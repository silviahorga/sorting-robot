/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * text_detect_cv_stage.cpp - Text Detector implementation, using OpenCV
 */
#include <iostream>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <libcamera/stream.h>
#include <memory>
#include <vector>

#include <libcamera/geometry.h>

#include "core/rpicam_app.hpp"

#include "post_processing_stages/post_processing_stage.hpp"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

using namespace cv;
using namespace cv::dnn;

using Stream = libcamera::Stream;

class TextDetectCvStage : public PostProcessingStage
{
public:
	TextDetectCvStage(RPiCamApp *app) : PostProcessingStage(app) {}

	char const *Name() const override;

	void Read(boost::property_tree::ptree const &params) override;

	void Configure() override;

	bool Process(CompletedRequestPtr &completed_request) override;

	void Stop() override;

private:
	void detectFeatures();
	void fourPointsTransform(const Mat &frame, const Point2f vertices[], Mat &result);
	void drawFeatures(cv::Mat &img);

	Stream *stream_;
	StreamInfo low_res_info_;
	Stream *full_stream_;
	StreamInfo full_stream_info_;
	std::unique_ptr<std::future<void>> future_ptr_;
	std::mutex text_mutex_;
	std::mutex future_ptr_mutex_;
	Mat image_;
	std::vector<std::vector<Point>> contours_;
	std::vector<std::string> text_;
	TextDetectionModel_DB detector_;
	TextRecognitionModel recognizer_;
	std::vector<std::string> vocabulary_;
	std::string detModelName_;
	std::string recModelName_;
	std::string vocName_;
	int refresh_rate_;
	int draw_features_;
};

#define NAME "text_detect_cv"

char const *TextDetectCvStage::Name() const
{
	return NAME;
}

void TextDetectCvStage::Read(boost::property_tree::ptree const &params)
{
	detModelName_ = params.get<char>("det_model_name", "/usr/local/share/OpenCV/db.pb");
	detector_ = TextDetectionModel_DB(detModelName_);

	recModelName_ = params.get<char>("rec_model_name", "/usr/local/share/OpenCV/crnn.onnx");
	recognizer_ = TextRecognitionModel(recModelName_);

	vocName_ = params.get<char>("vocabulary_name", "/usr/local/share/OpenCV/vocabulary.txt");
	refresh_rate_ = params.get<int>("refresh_rate", 5);
	draw_features_ = params.get<int>("draw_features", 1);
	int width = params.get<int>("width", 736);
	int height = params.get<int>("height", 736);
	float binThresh = params.get<float>("bin_thresh", 0.3);
	float polyThresh = params.get<float>("ply_thresh", 0.5);
	int maxCandidates = params.get<int>("max_candidates", 5);
	float unclipRatio = params.get<float>("unclip_ratio", 2.0);

	// Load detection model
	detector_.setBinaryThreshold(binThresh)
		.setPolygonThreshold(polyThresh)
		.setMaxCandidates(maxCandidates)
		.setUnclipRatio(unclipRatio);

	// Load vocabulary
	std::ifstream vocFile;
	vocFile.open(samples::findFile(vocName_));
	String vocLine;
	while (std::getline(vocFile, vocLine))
	{
		vocabulary_.push_back(vocLine);
	}
	recognizer_.setVocabulary(vocabulary_);
	recognizer_.setDecodeType("CTC-greedy");

	// Parameters for Recognition
	double recScale = 1.0 / 127.5;
	Scalar recMean = Scalar(127.5, 127.5, 127.5);
	Size recInputSize = Size(100, 32);
	recognizer_.setInputParams(recScale, recInputSize, recMean);

	// Parameters for Detection
	double scale = 1.0 / 255.0;
	Scalar mean = Scalar(122.67891434, 116.66876762, 104.00698793);
	Size inputSize = Size(width, height);
	detector_.setInputParams(scale, inputSize, mean);
}

void TextDetectCvStage::Configure()
{
	stream_ = nullptr;
	full_stream_ = nullptr;

	if (app_->StillStream()) // for stills capture, do nothing
		return;

	// Otherwise we expect there to be a lo res stream that we will use.
	stream_ = app_->LoresStream();
	if (!stream_)
		throw std::runtime_error("TextDetectCvStage: no low resolution stream");
	// (the lo res stream can only be YUV420)
	low_res_info_ = app_->GetStreamInfo(stream_);

	// We also expect there to be a "full resolution" stream which defines the output coordinate
	// system, and we can optionally draw the text there too.
	full_stream_ = app_->GetMainStream();
	if (!full_stream_)
		throw std::runtime_error("TextDetectCvStage: no full resolution stream available");
	full_stream_info_ = app_->GetStreamInfo(full_stream_);
	if (draw_features_ && full_stream_->configuration().pixelFormat != libcamera::formats::YUV420)
		throw std::runtime_error("TextDetectCvStage: drawing only supported for YUV420 images");
}

bool TextDetectCvStage::Process(CompletedRequestPtr &completed_request)
{
	if (!stream_)
		return false;

	{
		std::unique_lock<std::mutex> lck(future_ptr_mutex_);
		if (completed_request->sequence % refresh_rate_ == 0 &&
			(!future_ptr_ || future_ptr_->wait_for(std::chrono::seconds(0)) == std::future_status::ready))
		{
			if (future_ptr_ && future_ptr_->valid())
			{
				try
				{
					future_ptr_->get(); // This will rethrow the captured exception if any
				}
				catch (const std::exception &e)
				{
					std::cerr << "Error in detectFeatures: " << e.what() << std::endl;
				}
			}
			BufferReadSync r(app_, completed_request->buffers[stream_]);
			libcamera::Span<uint8_t> buffer = r.Get()[0];
			uint8_t *ptr = (uint8_t *)buffer.data();
			Mat image(low_res_info_.height, low_res_info_.width, CV_8U, ptr, low_res_info_.stride);
			image_ = image.clone();

			future_ptr_ = std::make_unique<std::future<void>>();
			*future_ptr_ = std::async(std::launch::async, [this] { detectFeatures(); });
		}
	}

	std::unique_lock<std::mutex> lock(text_mutex_);

	/*
	std::vector<libcamera::Rectangle> temprect;
	std::transform(faces_.begin(), faces_.end(), std::back_inserter(temprect),
				   [](Rect &r) { return libcamera::Rectangle(r.x, r.y, r.width, r.height); });
	completed_request->post_process_metadata.Set("detected_faces", temprect);
	 */

	if (draw_features_)
	{
		BufferWriteSync w(app_, completed_request->buffers[full_stream_]);
		libcamera::Span<uint8_t> buffer = w.Get()[0];
		uint8_t *ptr = (uint8_t *)buffer.data();
		Mat image(full_stream_info_.height, full_stream_info_.width, CV_8U, ptr, full_stream_info_.stride);
		drawFeatures(image);
	}

	return false;
}

void TextDetectCvStage::detectFeatures()
{
	std::vector<std::vector<Point>> detResults;
	Mat image_bgr;
	cvtColor(image_, image_bgr, COLOR_GRAY2BGR);
	detector_.detect(image_bgr, detResults);

	std::vector<std::vector<Point>> contours;
	std::vector<std::string> text;

	if (detResults.size() > 0)
	{
		for (uint i = 0; i < detResults.size(); i++)
		{
			const auto &quadrangle = detResults[i];
			CV_CheckEQ(quadrangle.size(), (size_t)4, "");

			contours.emplace_back(quadrangle);

			std::vector<Point2f> quadrangle_2f;
			for (int j = 0; j < 4; j++)
				quadrangle_2f.emplace_back(quadrangle[j]);

			Mat cropped;
			fourPointsTransform(image_, &quadrangle_2f[0], cropped);

			std::string recognitionResult = recognizer_.recognize(cropped);
			text.emplace_back(recognitionResult);
			std::cout << i << ": '" << recognitionResult << "'" << std::endl;
		}
	}

	// Scale contours back to the size and location in the full res image.
	double scale_x = full_stream_info_.width / (double)low_res_info_.width;
	double scale_y = full_stream_info_.height / (double)low_res_info_.height;
	for (auto &quadrangle : contours)
	{
		for (auto &point : quadrangle)
		{
			point.x *= scale_x;
			point.y *= scale_y;
		}
	}

	std::unique_lock<std::mutex> lock(text_mutex_);
	contours_ = std::move(contours);
	text_ = std::move(text);
}

void TextDetectCvStage::fourPointsTransform(const Mat &frame, const Point2f vertices[], Mat &result)
{
	float widthA = norm(vertices[0] - vertices[1]);
	float widthB = norm(vertices[2] - vertices[3]);
	float maxWidth = std::max(widthA, widthB);

	float heightA = norm(vertices[1] - vertices[2]);
	float heightB = norm(vertices[3] - vertices[0]);
	float maxHeight = std::max(heightA, heightB);

	const Size outputSize = Size(maxHeight, maxWidth);

	Point2f targetVertices[4] = { Point(0, outputSize.height - 1), Point(0, 0), Point(outputSize.width - 1, 0),
								  Point(outputSize.width - 1, outputSize.height - 1) };
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);

	warpPerspective(frame, result, rotationMatrix, outputSize);
}

void TextDetectCvStage::drawFeatures(Mat &img)
{
	// Draw text
	for (size_t i = 0; i < contours_.size(); i++)
	{
		const auto &quadrangle = contours_[i];
		const auto &text = text_[i];
		putText(img, text, quadrangle[3], FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
	}
	// Draw contours
	polylines(img, contours_, true, Scalar(0, 255, 0), 2);
}

void TextDetectCvStage::Stop()
{
	if (future_ptr_)
		future_ptr_->wait();
}

static PostProcessingStage *Create(RPiCamApp *app)
{
	return new TextDetectCvStage(app);
}

static RegisterStage reg(NAME, &Create);
