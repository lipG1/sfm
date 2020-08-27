#include "_2_stages_stereo.h"


void stereo2::extractFeatures(const Mat& img, const Mat& img1)
{
	resize(img, img, Size(), downscaleFactor, downscaleFactor);
	resize(img1, img1, Size(), downscaleFactor, downscaleFactor);
	Ptr<SURF> surf = SURF::create();
	Mat mask;
	surf->detectAndCompute(img, mask,leftFeatures.keyPoints,leftFeatures.descriptors);
	surf->detectAndCompute(img1, mask,rightFeatures.keyPoints, rightFeatures.descriptors);
}

void stereo2::matchFeatures()
{
	vector<vector<DMatch>> matches;
	vector<DMatch> match_good;
	FlannBasedMatcher match;
	Mat mask;
	match.knnMatch(leftFeatures.descriptors, rightFeatures.descriptors, matches ,2,mask);
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i].size() < 2)
			continue;
		const DMatch& m0 = matches[i][0];//left->right
		const DMatch& m1 = matches[i][1];//right->left
		if (m0.distance < 0.6 * m1.distance)
			match_good.push_back(m0);
	}
	Mat objectPoints(1, static_cast<int>(match_good.size()), CV_32FC2);
	Mat scenePoints(1, static_cast<int>(match_good.size()), CV_32FC2);
	for (size_t i = 0; i < match_good.size(); i++)
	{
		const DMatch& m = match_good[i];
		Point2f p = leftFeatures.keyPoints[m.queryIdx].pt;
		objectPoints.at<Point2f>(static_cast<int>(i)) = p;
		 p = rightFeatures.keyPoints[m.trainIdx].pt;
		scenePoints.at<Point2f>(static_cast<int>(i)) = p;
	}
	leftFinalPoints  = objectPoints.clone();
	rightFinalPoints = scenePoints.clone();
	
}

bool stereo2::findCameraMatricesFromMatch(const Mat& K, Matx34f& Pleft, Matx34f& Pright)
{
	if (K.empty())
	{
		cout << "The K must be initalized! ";//the K must be initalized！
		return false;
	}
	else
	{
		Mat E,R,T,mask;
		E = findEssentialMat(leftFinalPoints, rightFinalPoints, K, 8, 0.999, 1.0, mask);
		recoverPose(E, leftFinalPoints, rightFinalPoints, K, R, T, mask);
		Pleft = Matx34f::eye();
		Pright = Matx34f(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
			             R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
			             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2));
		
		int size_ = 0;
		for (int i = 0; i < mask.rows; i++)
		{
			if (mask.at<uchar>(i))
				size_++;
		}
		Mat leftPruned = Mat(1, static_cast<int>(size_), CV_32FC2);
		Mat rightPruned = Mat(1, static_cast<int>(size_), CV_32FC2);
		int a = 0;
		for (int i = 0; i < mask.rows; i++)
		{
			if (mask.at<uchar>(i))
			{
				Point2f p = leftFinalPoints.at<Point2f>(static_cast<int>(i));
				leftPruned.at<Point2f>(static_cast<int>(a)) = p;
				p = rightFinalPoints.at<Point2f>(i);
				rightPruned.at<Point2f>(static_cast<int>(a)) = p;
				a++;
			}
		}
		prunedPointsLeft = leftPruned.clone();
		prunedPointsRight = rightPruned.clone();

		return true;
	}

}

void stereo2::BAestimate()
{
	//camera parameter need to be initialized before！
	CvLevMarq LM(12, 2 * prunedPointsLeft.cols, cvTermCriteria(term_criteria_));
	Mat _err, _jac;//LM Mat
	CvMat camera_parma = cvMat(cam_param);//using CvMat
	cvCopy(&camera_parma, LM.param);//LM get the camera_para

	int iter = 0;
	for (;;)
	{
		const CvMat*  parma = 0;// rename the address
		CvMat*  _jac_ = 0;
		CvMat*  _err_ = 0;
		

		bool proceed = LM.update(parma, _jac_, _err_);
		cvCopy(parma, &camera_parma);

		if (!proceed || !_err_)
			break;//end the LM

		if (_jac_)//need to update the jac Matrix
		{
			calJacobian(_jac);
			CvMat tep = _jac;
			cvCopy(&tep, _jac_);
		}
		if (_err_)//need to update the error Matrix
		{
			triangluate(_err);
			iter++;
			CvMat tmp = cvMat(_err);
			cvCopy(&tmp, _err_);
		}

		   
	}

}

void stereo2::triangluate(Mat& err)
{
	Mat normalizedLeftPts, normalizedRightPts;
	undistortPoints(prunedPointsLeft,normalizedLeftPts,K,Mat());
	undistortPoints(prunedPointsRight, normalizedRightPts, K, Mat());
	Mat recvLeft(3,1,CV_64F),Rleft(3, 3, CV_64F);
	double rp = cam_param.at<double>(0, 0);
	recvLeft.at<double>(0, 0) =rp;
	rp = cam_param.at<double>(1, 0);
	recvLeft.at<double>(1, 0) = rp;
	rp = cam_param.at<double>(2, 0);
	recvLeft.at<double>(2, 0) = rp;
	Rodrigues(recvLeft, Rleft);//we need the 3x3 R
	
	Mat tecvLeft(3,1,CV_64F);//get the T Matrix
	rp = cam_param.at<double>(3, 0);
	tecvLeft.at<double>(0, 0) = rp;
	rp = cam_param.at<double>(4, 0);
	tecvLeft.at<double>(1, 0) = rp;
	rp = cam_param.at<double>(5, 0);
	tecvLeft.at<double>(2, 0) = rp;
	Matx34f Pleft;
	//Pleft must be created before,we cannot create it like Matx Pleft = Matx () ;
	Pleft=Matx34f(Rleft.at<double>(0,0), Rleft.at<double>(0, 1), Rleft.at<double>(0, 2), tecvLeft.at<double>(0, 0),
		Rleft.at<double>(1, 0), Rleft.at<double>(1, 1), Rleft.at<double>(1, 2), tecvLeft.at<double>(1, 0),
		Rleft.at<double>(2, 0), Rleft.at<double>(2, 1), Rleft.at<double>(2, 2), tecvLeft.at<double>(2, 0));

	Mat recvRight(3, 1, CV_64F), Rright(3, 3, CV_64F);
	rp = cam_param.at<double>(6, 0);
	recvRight.at<double>(0, 0) = rp;
	rp = cam_param.at<double>(7, 0);
	recvRight.at<double>(1, 0) = rp;
	rp = cam_param.at<double>(8, 0);
	recvRight.at<double>(2, 0) = rp;
	Rodrigues(recvRight, Rright);//

	Mat tecvRight(3, 1, CV_64F);//
	rp = cam_param.at<double>(9, 0);
	tecvRight.at<double>(0, 0) = rp;
	rp = cam_param.at<double>(10, 0);
	tecvRight.at<double>(1, 0) = rp;
	rp = cam_param.at<double>(11, 0);
	tecvRight.at<double>(2, 0) = rp;
	
	Matx34f Pright;
	Pright = Matx34f(Rright.at<double>(0, 0), Rright.at<double>(0, 1), Rright.at<double>(0, 2), tecvRight.at<double>(0, 0),
		Rright.at<double>(1, 0), Rright.at<double>(1, 1), Rright.at<double>(1, 2), tecvRight.at<double>(1, 0),
		Rright.at<double>(2, 0), Rright.at<double>(2, 1), Rright.at<double>(2, 2), tecvRight.at<double>(2, 0));
	
	Mat homogeneous3d;
	triangulatePoints(Pleft, Pright, normalizedLeftPts, normalizedRightPts, homogeneous3d);//get the 4x1 points
	
	Mat Points_3d;
	convertPointsFromHomogeneous(homogeneous3d.t(), Points_3d);//change 4x1 points to 3d points;
	err.create(2 * prunedPointsLeft.cols, 1, CV_64F);
	
	
	vector<Point2f> projectedLeftToRight(prunedPointsLeft.cols);
	projectPoints(Points_3d, recvRight, tecvRight.t(), K, Mat(), projectedLeftToRight);//left -> right
	vector<Point3f> none;
	points3d.swap(none);
	for (int i = 0; i < projectedLeftToRight.size(); i++)//
	{
		Point2f error = prunedPointsRight.at<Point2f>(static_cast<int>(i))-projectedLeftToRight[i];
		err.at<double>(i * 2, 0) = error.x;
		err.at<double>(i * 2+1, 0) = error.y;
		Point3f p;
		p = Points_3d.at<Point3f>(static_cast<int>(i));
		points3d.push_back(p);
	}

	//Mat recvRight;
	//Rodrigues(Pright.get_minor<3, 3>(0, 0), recvRight);//ת����R����
	//Mat tecvRight(Pright.get_minor<3, 1>(0, 3).t());//T����
	//vector<Point2f> projectedRight(prunedPointsRight.rows);
	//projectPoints(Points_3d, recvRight, tecvRight, K, Mat(), projectedRight);//�ҵ�
	//for (int i = 0; i < Points_3d.rows; i++)//����ά�����ɸѡ
	//{
	//	int index = prunedPointsLeft.rows * 2+ 2 * i;
	//	Point2f error = prunedPointsRight.at<Point2f>(i)-projectedRight[i];
	//	err.at<double>(index, 0) = abs(error.x);
	//	err.at<double>(1+index, 1) = abs(error.y);
	//	
	//}
	//for (int i = 0; i < Points_3d.rows; i++)//����ά�����ɸѡ
	//{
	//	if (norm(projectedLeft[i] - prunedPointsLeft.at<Point2f>(i)) > MIN_REPROJECTION_ERROR
	//		or norm(projectedRight[i] - prunedPointsRight.at<Point2f>(i)) > MIN_REPROJECTION_ERROR)
	//	{
	//		continue;
	//	}
	//	Point3f p;
	//	p = Point3f(Points_3d.at<float>(i, 0),
	//		Points_3d.at<float>(i, 1),
	//		Points_3d.at<float>(i, 2));
	//	points3d.push_back(p);
	//}
	//
	
}

void stereo2::calDeriv(Mat& err1, Mat& err2, double h, Mat& res)
{
	//����΢��
	for (int i = 0; i < err1.rows; i++)
		res.at<double>(i, 0) = (err1.at<double>(i, 0) - err2.at<double>(i, 0)) / h;
}

void stereo2::calJacobian(Mat& jac)
{
	jac.create(2 * prunedPointsLeft.cols, 12, CV_64F);
	jac.setTo(0);
	double val;
	const double step = 1e-4;//the step 
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 6; j++)//get the gridents
		{
			val = cam_param.at<double>(i*6+j, 0);
			cam_param.at<double>(i*6+j, 0) = val - step;
			triangluate(err1);
			cam_param.at<double>(i*6+j, 0) = val + step;
			triangluate(err2);
			Mat col_n = jac.col(i*6 + j);
			calDeriv(err1, err2, 2 * step, col_n);
			cam_param.at<double>(i * 6 + j, 0) = val;
		}
	}
}

void stereo2::setUpIntitalCameraPara(const Matx34f& Pleft, const Matx34f& Pright)
{
	cam_param.create(12,1, CV_64F);
	// get the orthogonality matrix of R
	SVD svd;
	Mat R(Pleft.get_minor<3,3>(0,0));
	svd(R,SVD::FULL_UV);
	Mat R1;
	R1 = svd.u * svd.vt;
	if (determinant(R1) < 0)
		R1 *= -1;
	Mat recv1;
	Rodrigues(R1, recv1);
	CV_Assert(recv1.type() == CV_32F);
	float pam = recv1.at<float>(0, 0);
	cam_param.at<double>(0, 0) = pam;//pam as the temp to copy the left variable to right variable
	pam = recv1.at<float>(1, 0);
	cam_param.at<double>(1, 0) = pam;
	pam = recv1.at<float>(2, 0);
	cam_param.at<double>(2, 0) = pam;
	Mat T1(Pleft.get_minor<3, 1>(0, 3));
	pam = T1.at<float>(0, 0);
	cam_param.at<double>(3, 0) = pam;
	pam = T1.at<float>(1, 0);
	cam_param.at<double>(4, 0) = pam;
	pam = T1.at<float>(2, 0);
	cam_param.at<double>(5, 0) = pam;
	Mat R2(Pright.get_minor<3, 3>(0, 0));
	svd(R2, SVD::FULL_UV);
	Mat R3;
	R3 = svd.u * svd.vt;
	if (determinant(R3) < 0)
		R3 *= -1;
	Mat recv2;
	Rodrigues(R3, recv2);
	CV_Assert(recv2.type() == CV_32F);
	pam = recv2.at<float>(0, 0);
	cam_param.at<double>(6, 0) = pam;
	pam = recv2.at<float>(1, 0);
	cam_param.at<double>(7, 0) = pam;
	pam = recv2.at<float>(2, 0);
	cam_param.at<double>(8, 0) = pam;
	Mat tecv2(Pright.get_minor<3, 1>(0, 3));
	pam = tecv2.at<float>(0, 0);
	cam_param.at<double>(9, 0) = pam;
	pam = tecv2.at<float>(1, 0);
	cam_param.at<double>(10, 0) = pam;
	pam = tecv2.at<float>(2, 0);
	cam_param.at<double>(11, 0) = pam;


}


//bool stereo2::bundleAdjustment( const Mat& K, const vector<Point3f>& points3d)
//{
//	// Create residuals for each observation in the bundle adjustment problem. The
//   // parameters for cameras and points are added automatically.
//	ceres::Problem Problem;
//	//Convert camera pose parameters from [R|t] (3x4) to [Angle-Axis (3), Translation (3), focal (1)] (1x7)
//	typedef cv::Matx<double, 1, 6> CameraVector;
//	vector<CameraVector> cameraPoses6d;
//	cameraPoses6d.reserve(pose.size());
//	for (int i = 0; i < pose.size(); i++)
//	{
//		const Matx34f po = pose[i];
//		Vec3f T = (po(0, 3), po(1, 3), po(2, 3));
//		Matx33f R = po.get_minor<3, 3>(0, 0);
//		float angleAxis[3];
//		ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis);//Ceres assumes col-major...
//
//		cameraPoses6d.push_back(CameraVector(
//			angleAxis[0],
//			angleAxis[1],
//			angleAxis[2],
//			T(0),
//			T(1),
//			T(2)));
//	}
//	double focal = K.at<double>(0, 0);
//	vector<Vec3d> point3d(points3d.size());
//	for (int i = 0; i < points3d.size(); i++)
//	{
//		Point3f point = points3d[i];
//		point3d[i] = Vec3d(point.x, point.y, point.z);
//		for (int i = 0; i < prunedPointsLeft.rows; i++)
//		{
//			Point2f p2d = prunedPointsLeft.at<Point2f>(i);
//			p2d.x -= K.at<double>(0, 2);
//			p2d.y -= K.at<double>(1, 2);
//			// Each Residual block takes a point and a camera as input and outputs a 2
//			// dimensional residual. Internally, the cost function stores the observed
//			// image location and compares the reprojection against the observation.
//			ceres::CostFunction* cost_function = SimpleReprojectionError::Create(p2d.x, p2d.y);
//
//			Problem.AddResidualBlock(cost_function,
//				NULL /* squared loss */,
//				cameraPoses6d[0].val,
//				point3d[i].val,
//				&focal);
//		}
//		for (int i = 0; i < prunedPointsRight.rows; i++)
//		{
//			Point2f p2d = prunedPointsRight.at<Point2f>(i);
//			p2d.x -= K.at<double>(0, 2);
//			p2d.y -= K.at<double>(1, 2);
//			// Each Residual block takes a point and a camera as input and outputs a 2
//			// dimensional residual. Internally, the cost function stores the observed
//			// image location and compares the reprojection against the observation.
//			ceres::CostFunction* cost_function = SimpleReprojectionError::Create(p2d.x, p2d.y);
//
//			Problem.AddResidualBlock(cost_function,
//				NULL /* squared loss */,
//				cameraPoses6d[1].val,
//				point3d[i].val,
//				&focal);
//		}
//	}
//	// Make Ceres automatically detect the bundle structure. Note that the
//	// standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
//	// for standard bundle adjustment problems.
//	ceres::Solver::Options options;
//	options.linear_solver_type = ceres::DENSE_SCHUR;
//	options.minimizer_progress_to_stdout = true;
//	options.max_num_iterations = 500;
//	options.eta = 1e-2;
//	options.max_solver_time_in_seconds = 10;
//	options.logging_type = ceres::LoggingType::SILENT;
//	ceres::Solver::Summary summary;
//	ceres::Solve(options, &Problem, &summary);
//	std::cout << summary.BriefReport() << "\n";
//
//	if (not (summary.termination_type == ceres::CONVERGENCE)) {
//		cerr << "Bundle adjustment failed." << endl;
//		return false;
//	}
//
//
//
//	//Implement the optimized camera poses and 3D points back into the reconstruction
//	for (size_t i = 0; i < pose.size(); i++)
//	{
//		Matx34f po=pose[i];
//		//Convert optimized Angle-Axis back to rotation matrix
//		double rotationMat[9] = { 0 };
//		ceres::AngleAxisToRotationMatrix(cameraPoses6d[i].val, rotationMat);
//
//		for (int r = 0; r < 3; r++) {
//			for (int c = 0; c < 3; c++) {
//				po(c, r) = rotationMat[r * 3 + c]; //`rotationMat` is col-major...
//			}
//		}
//
//		//Translation
//		po(0, 3) = cameraPoses6d[i](3);
//		po(1, 3) = cameraPoses6d[i](4);
//		po(2, 3) = cameraPoses6d[i](5);
//	}
//	vector<Point3f> point_copy(points3d.size());
//	for (int i = 0; i < points3d.size(); i++) {
//		Point3f points;
//		points = Point3f(point3d[i](0), point3d[i](1), point3d[i](2));
//		point_copy.push_back(points);
//	}
//
//}


