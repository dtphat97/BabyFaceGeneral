#include "stdafx.h"
#include "HeaderTool.h"

// hàm kiểm tra và nhận diện khuôn mặt
int getFeaturePoints(std::vector<Point2i> &PointList, string ImgName) {
	string DataSet = "machinelearning.dat";

	try
	{
		/// Sử dụng hàm của thư viện dlib để phát hiện khuôn mặt
		frontal_face_detector DetectorFrontal = get_frontal_face_detector();
		shape_predictor ShapePredictor;
		deserialize(DataSet) >> ShapePredictor;

		cout << "Processing feature points: " << ImgName << endl;
		array2d<rgb_pixel> ImageProc;
		load_image(ImageProc, ImgName);
	
		
		std::vector<dlib::rectangle> Faces = DetectorFrontal(ImageProc);
		cout << "\n\tDetect: " << Faces.size() << " (face)" << endl;
		//kiểm tra có duy nhât một khuôn mặt trong ảnh hay không
		if (Faces.size() != 1) {
			cout << "\n\tExit!";
			return -1;
		}

		full_object_detection ShapeFace = ShapePredictor(ImageProc, Faces[0]);
		cout << "\n\twaiting for a few minutes..." << endl;
		// lưu vector của khuôn mặt vào danh sách
		for (int h = 0; h < ShapeFace.num_parts(); h++) {
			
			Point2i ThisPoint(ShapeFace.part(h).x(), ShapeFace.part(h).y());
			PointList.push_back(ThisPoint);
		}

	}
	catch (exception& e) {
		cout << "\nError: " << endl;
		cout << e.what() << endl;
	}
}


// Sau khi có danh sach vector điểm mặt, còn thiếu 11 vector nữa mới đủ 80 , nên viết hàm phát sinh thêm
void getMorePoints(std::vector<Point2i> &PointList) {
	int DeltaY = PointList[36].y - PointList[18].y;;

	PointList.push_back(Point2i(PointList[0].x, PointList[0].y - DeltaY));

	for (int i = 16; i < 28; i++) {
		PointList.push_back(Point2i(PointList[i].x, PointList[i].y - DeltaY));
	}
}

void morphBabyFromParents(Mat &Father, Mat &Mother, Mat &MorphImage, std::vector<Point2f> &TriangleFather, std::vector<Point2f> &TriangleMother, std::vector<Point2f> &TriangleMorph, double Alpha) {
	
	Rect RectMorph = boundingRect(TriangleMorph);
	Rect RectFather = boundingRect(TriangleFather);
	Rect RectMother = boundingRect(TriangleMother);


	std::vector<Point2f> PointsFather, PointsMother, PointsMorph;
	std::vector<Point> PointsIntMorph;
	for (int i = 0; i < 3; i++) {
		PointsMorph.push_back(Point2f(TriangleMorph[i].x - RectMorph.x, TriangleMorph[i].y - RectMorph.y));
		PointsIntMorph.push_back(Point(TriangleMorph[i].x - RectMorph.x, TriangleMorph[i].y - RectMorph.y));
		PointsFather.push_back(Point2f(TriangleFather[i].x - RectFather.x, TriangleFather[i].y - RectFather.y));
		PointsMother.push_back(Point2f(TriangleMother[i].x - RectMother.x, TriangleMother[i].y - RectMother.y));
	}

	/// Tạo mặt nạ
	Mat Mask = Mat::zeros(RectMorph.height, RectMorph.width, CV_32FC3);
	/// Fill Triangle
	fillConvexPoly(Mask, PointsIntMorph, Scalar(1.0, 1.0, 1.0), 16, 0);

	
	Mat ROIFather, ROIMother;
	Father(RectFather).copyTo(ROIFather);
	Mother(RectMother).copyTo(ROIMother);

	Mat PreAlphaFather = Mat::zeros(RectMorph.height, RectMorph.width, ROIFather.type());
	Mat PreAlphaMother = Mat::zeros(RectMorph.height, RectMorph.width, ROIMother.type());

	/// Biến đổi ảnh
	Mat WrapMatFather = getAffineTransform(PointsFather, PointsMorph);
	warpAffine(ROIFather, PreAlphaFather, WrapMatFather, PreAlphaFather.size(), INTER_LINEAR, BORDER_REFLECT_101);
	/// Biến đổi ảnh
	Mat WrapMatMother = getAffineTransform(PointsMother, PointsMorph);
	warpAffine(ROIMother, PreAlphaMother, WrapMatMother, PreAlphaMother.size(), INTER_LINEAR, BORDER_REFLECT_101);

	// trộn hình lại theo hệ số alpha
	Mat AlphaImage = Alpha * PreAlphaFather + (1.0 - Alpha) * PreAlphaMother;

	/// Sao chép ảnh
	multiply(AlphaImage, Mask, AlphaImage);
	multiply(MorphImage(RectMorph), Scalar(1.0, 1.0, 1.0) - Mask, MorphImage(RectMorph));
	MorphImage(RectMorph) = MorphImage(RectMorph) + AlphaImage;
}