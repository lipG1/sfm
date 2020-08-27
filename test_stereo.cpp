#include "_2_stages_stereo.h"


int main()
{
	Mat img1 = imread("E:\\pic\\stereo\\left.jpg");
	Mat img2 = imread("E:\\pic\\stereo\\right.jpg");
	Mat K(3,3,CV_64F);
	K.at<double>(0, 0) = 2493.57;
	K.at<double>(0, 1) = 0;
	K.at<double>(0, 2) = 1284.1;
	K.at<double>(1, 0) = 0;
	K.at<double>(1, 1) = 2494.61;
	K.at<double>(1, 2) = 1035.04;
	K.at<double>(2, 0) = 0;
	K.at<double>(2, 1) = 0;
	K.at<double>(2, 2) = 1;
	stereo2 st(1.0,K);
	st.extractFeatures(img1, img2);
	st.matchFeatures();
	Matx34f Pleft, Pright;
	bool yes=st.findCameraMatricesFromMatch(K, Pleft, Pright);
	if (!yes)
	{
		cout << "cannot get the right matches!";
	}
	st.setUpIntitalCameraPara(Pleft, Pright);
	st.BAestimate();
	vector<Point3f> points = st.points3d;
	for (int i = 0; i < points.size(); i++)
	{
		cout << points[i].x << " " << points[i].y << " " << points[i].z << " "<<endl;
	}
	
	

}
