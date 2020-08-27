#pragma once
#ifndef _2_stages_stereo
#define _2_stages_stereo

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/stitching/detail/motion_estimators.hpp>
#include<iostream>
#include<thread>
#include<mutex>
//#include<ceres/ceres.h>
//#include<ceres/rotation.h>

using namespace cv;
using namespace xfeatures2d;
using namespace std;



struct Features
{
	vector<KeyPoint> keyPoints;
	Mat            descriptors;
};

	class stereo2
	{
	public:
		
		stereo2(const float downscale ,const Mat& k1) 
		{
			downscaleFactor = downscale;
		    setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 1000, DBL_EPSILON));
			K = k1;
		};//Ҳ����ֱ�ӳ�ʼ���ǵü�ð��
		
		void extractFeatures(const Mat& img, const Mat& img1);
		void matchFeatures();
		bool findCameraMatricesFromMatch(const Mat& K,  Matx34f& Pleft, Matx34f& Pright);
		void setUpIntitalCameraPara(const Matx34f& Pleft, const Matx34f& Pright);
		//bool bundleAdjustment( const Mat& K, const vector<Point3f>& points3d);
		void BAestimate();
		vector<Point3f> points3d;
		TermCriteria termCriteria() { return term_criteria_; }
		void setTermCriteria(const TermCriteria& term_criteria) { term_criteria_ = term_criteria; }
		~stereo2() {};//������������д�����������ʲô��û�У�һ��Ҫһһ��Ӧ
	private:
		float downscaleFactor;
		Features leftFeatures;
		Features rightFeatures;
		Mat leftFinalPoints;
		Mat rightFinalPoints;
		Mat prunedPointsLeft;
		Mat prunedPointsRight;
		Mat cam_param;
		Mat err1,err2;
		Mat K;
		
		TermCriteria term_criteria_;
		const float MIN_REPROJECTION_ERROR = 10.0;
		void triangluate(Mat& err);
		void calDeriv(Mat& err1, Mat& err2, double h, Mat& res);
		void calJacobian(Mat& jac);
		//struct SimpleReprojectionError
		//{
		//	SimpleReprojectionError(double observed_x, double observed_y) :
		//		observed_x(observed_x), observed_y(observed_y) {
		//	}
		//	template<typename T>
		//	bool operator()(const T* const camera,
		//		const T* const point,
		//		const T* const focal,
		//		T* residuals) const {
		//		T p[3];
		//		// Rotate: camera[0,1,2] are the angle-axis rotation.
		//		ceres::AngleAxisRotatePoint(camera, point, p);

		//		// Translate: camera[3,4,5] are the translation.
		//		p[0] += camera[3];
		//		p[1] += camera[4];
		//		p[2] += camera[5];

		//		// Perspective divide
		//		const T xp = p[0] / p[2];
		//		const T yp = p[1] / p[2];

		//		// Compute final projected point position.
		//		const T predicted_x = *focal * xp;
		//		const T predicted_y = *focal * yp;

		//		// The error is the difference between the predicted and observed position.
		//		residuals[0] = predicted_x - T(observed_x);
		//		residuals[1] = predicted_y - T(observed_y);
		//		return true;
		//	}
		//	// Factory to hide the construction of the CostFunction object from
		//	// the client code.
		//	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		//		return (new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
		//			new SimpleReprojectionError(observed_x, observed_y)));
		//	}
		//	double observed_x;
		//	double observed_y;
		//};
	};

#endif // !_2_stages_stereo

