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
	TextDetectionModel_EAST detector_;
	TextRecognitionModel recognizer_;
	std::vector<std::string> vocabulary_;
	std::string detModelName_;
	std::string recModelName_;
	std::string vocName_;
	double scaling_factor_;
	int min_neighbors_;
	int min_size_;
	int max_size_;
	int refresh_rate_;
	int draw_features_;
	int width_;
	int height_;
	float conf_threshold_;
	float nms_threshold_;
};

#define NAME "text_detect_cv"

char const *TextDetectCvStage::Name() const
{
	return NAME;
}

void TextDetectCvStage::Read(boost::property_tree::ptree const &params)
{
	detModelName_ = params.get<char>("det_model_name", "/usr/local/share/OpenCV/east.pb");
	detector_ = TextDetectionModel_EAST(detModelName_);

	recModelName_ = params.get<char>("rec_model_name", "/usr/local/share/OpenCV/crnn.onnx");
	recognizer_ = TextRecognitionModel(recModelName_);

	vocName_ = params.get<char>("vocabulary_name", "/usr/local/share/OpenCV/vocabulary.txt");
	scaling_factor_ = params.get<double>("scaling_factor", 1.1);
	min_neighbors_ = params.get<int>("min_neighbors", 3);
	min_size_ = params.get<int>("min_size", 32);
	max_size_ = params.get<int>("max_size", 256);
	refresh_rate_ = params.get<int>("refresh_rate", 5);
	draw_features_ = params.get<int>("draw_features", 1);
	width_ = params.get<int>("width", 320);
	height_ = params.get<int>("height", 320);
	conf_threshold_ = params.get<float>("conf_threshold", 0.5);
	nms_threshold_ = params.get<float>("nms_threshold", 0.4);

	detector_.setConfidenceThreshold(conf_threshold_).setNMSThreshold(nms_threshold_);

	std::ifstream vocFile;
	vocFile.open(samples::findFile(vocName_));
	String vocLine;
	while (std::getline(vocFile, vocLine))
	{
		vocabulary_.push_back(vocLine);
	}
	recognizer_.setVocabulary(vocabulary_);
	recognizer_.setDecodeType("CTC-greedy");

	double recScale = 1.0 / 127.5;
	Scalar recMean = Scalar(127.5, 127.5, 127.5);
	Size recInputSize = Size(100, 32);
	recognizer_.setInputParams(recScale, recInputSize, recMean);

	// Parameters for Detecction
	double detScale = 1.0;
	Size detInputSize = Size(width_, height_);
	Scalar detMean = Scalar(123.68, 116.78, 103.94);
	bool swapRB = true;
	detector_.setInputParams(detScale, detInputSize, detMean, swapRB);
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
			Mat image(low_res_info_.height, low_res_info_.width, CV_8UC3, ptr, low_res_info_.stride * 3);
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
	detector_.detect(image_, detResults);

	cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
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

	std::unique_lock<std::mutex> lock(text_mutex_);
	contours_ = std::move(contours);
	text_ = std::move(text);
}

void TextDetectCvStage::fourPointsTransform(const Mat &frame, const Point2f vertices[], Mat &result)
{
	const Size outputSize = Size(100, 32);

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
