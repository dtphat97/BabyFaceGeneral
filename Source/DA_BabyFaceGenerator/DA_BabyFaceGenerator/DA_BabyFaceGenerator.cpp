// DA_BabyFaceGenerator.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "HeaderTool.h"


void main()
{
	/// Khung sường khuôn mặt đứa bé
	string ImgChild = "baby.jpg";
	std::vector<Vec6f> ChildTriList;

	cout << "\n\t-------------------- Processing --------------------";
	/// Đoc ảnh CV_32F
	Mat ChildImage = imread(ImgChild);	

	ChildImage.convertTo(ChildImage, CV_32F);

	

	/// Đưa tọa độ điểm khuôn mặt vào danh sách, 69 điểm
	std::vector<Point2i> ChildPoint;

	cout << "\n\t  Please wait...";
	if (getFeaturePoints(ChildPoint, ImgChild) == -1) {
		cout << "\n\tError: detector gets more than 2 faces.";
		cin.get();
		return;
	}
	cout << endl;
	
	// Phát sinh thêm tập dữ liệu  thêm vào danh sách các tọa độ điểm trên khuôn mặt
	getMorePoints(ChildPoint);

	///---------------------- Father ----------------------
	string ImgFather;
	std::vector<Vec6f> FaTriList;

	/// Nhập ảnh bố
	cout << "\n\t---------------------- Father ----------------------";
	cout << "\n\t  Input: ";
	getline(cin, ImgFather);

	/// Đoc ảnh CV_32F
	Mat FaImage = imread(ImgFather);
	FaImage.convertTo(FaImage, CV_32F);

	/// Đưa tọa độ điểm khuôn mặt vào danh sách, 69 điểm
	std::vector<Point2i> FatherPoint;

	cout << "\n\t  Please wait...";
	if (getFeaturePoints(FatherPoint, ImgFather) == -1) {
		cout << "\n\tError: detector gets more than 2 faces.";
		cin.get();
		return;
	}

	///---------------------- Mother ----------------------
	string ImgMother;
	std::vector<Vec6f> MoTriList;

	cout << "\n\t---------------------- Mother ----------------------";
	cout << "\n\t  Input: ";
	getline(cin, ImgMother);

	///  Nhập ảnh mẹ
	Mat MoImage = imread(ImgMother);
	MoImage.convertTo(MoImage, CV_32F);

	/// Đưa tọa độ điểm khuôn mặt vào danh sách, 69 điểm
	std::vector<Point2i> MotherPoint;

	cout << "\n\t  Please wait...";
	if (getFeaturePoints(MotherPoint, ImgMother) == -1) {
		cout << "\n\tError: detector gets more than 2 faces.";
		cin.get();
		return;
	}


	// Phát sinh thêm tập dữ liệu  thêm vào danh sách các tọa độ điểm trên khuôn mặt bố
	getMorePoints(FatherPoint);
	// Phát sinh thêm tập dữ liệu  thêm vào danh sách các tọa độ điểm trên khuôn mặt mẹ
	getMorePoints(MotherPoint);
	
	/// Phép đổi hình ảnh
	Mat ImgMorph = Mat::zeros(FaImage.size(), CV_32FC3);
	std::vector<Point2f> Points;

	cout << "\n\t----------------------  Baby  ----------------------";

	cout << "\n\tInput alpha: \n";
	double Alpha = 0.5;

	do {
		cout << "\n\tInput: ";
		cin >> Alpha;
	} while (Alpha <= 0 || Alpha >= 1);


	///Tính xấp xỉ giống nhau với bố và mẹ
	for (int i = 0; i < FatherPoint.size(); i++) {
		float X, Y;
		X = Alpha * FatherPoint[i].x + (1 - Alpha) * MotherPoint[i].x;
		Y = Alpha * FatherPoint[i].y + (1 - Alpha) * MotherPoint[i].y;

		// push point
		Points.push_back(Point2f(X, Y));
	}


	/// Đọc 3 điểm trong file Triangulation.txt để biến đổi vector khuôn mặt của bố, mẹ và ảnh con ban đầu
	ifstream FileTri("Triangulation.txt");

	/// Hàm biến đổi phát sinh ảnh  khuôn mặt của đứa con khi lớn
	int Vertex1, Vertex2, Vertex3;

	while (FileTri >> Vertex1 >> Vertex2 >> Vertex3) {
		///  danh sách các vector mới 
		std::vector<Point2f> TriangleFather, TriangleMother, TriangleMorph;

		//  vùng ảnh khuôn mặt của bố
		TriangleFather.push_back(FatherPoint[Vertex1]);
		TriangleFather.push_back(FatherPoint[Vertex2]);
		TriangleFather.push_back(FatherPoint[Vertex3]);

		// vùng ảnh khuôn mặt của mẹ
		TriangleMother.push_back(MotherPoint[Vertex1]);
		TriangleMother.push_back(MotherPoint[Vertex2]);
		TriangleMother.push_back(MotherPoint[Vertex3]);
		// vùng ảnh khuôn mặt đưa con khi lớn
		TriangleMorph.push_back(Points[Vertex1]);
		TriangleMorph.push_back(Points[Vertex2]);
		TriangleMorph.push_back(Points[Vertex3]);

		/// Phát sinh ảnh từ bố mẹ
		morphBabyFromParents(FaImage, MoImage, ImgMorph, TriangleFather, TriangleMother, TriangleMorph, Alpha);
	}




	///---------------------- Morphing ----------------------
	Mat ImgBaby = Mat::zeros(FaImage.size(), CV_32FC3);

	/// Close file
	FileTri.close();

	
	/// Hàm biến đổi phát sinh ảnh  khuôn mặt của đứa con khi lớn
	ifstream FileTriMini("Triangulation.txt");

	while (FileTriMini >> Vertex1 >> Vertex2 >> Vertex3) {
		///  danh sách các vector mới
		std::vector<Point2f> TriangleFather, TriangleMother, TriangleMorph;

		/// vùng ảnh khuôn mặt của bố
		TriangleFather.push_back(FatherPoint[Vertex1]);
		TriangleFather.push_back(FatherPoint[Vertex2]);
		TriangleFather.push_back(FatherPoint[Vertex3]);

		///vùng ảnh khuôn mặt của mẹ
		TriangleMother.push_back(MotherPoint[Vertex1]);
		TriangleMother.push_back(MotherPoint[Vertex2]);
		TriangleMother.push_back(MotherPoint[Vertex3]);

		/// vùng ảnh khuôn mặt của con
		TriangleMorph.push_back(ChildPoint[Vertex1]);
		TriangleMorph.push_back(ChildPoint[Vertex2]);
		TriangleMorph.push_back(ChildPoint[Vertex3]);

		/// Phát sinh ảnh từ bố mẹ
		morphBabyFromParents(FaImage, MoImage, ImgBaby, TriangleFather, TriangleMother, TriangleMorph, Alpha);
	}



	FileTriMini.close();

	/// Hiển thị ảnh bố
	namedWindow("Father - BabyFaceGenerator", WINDOW_NORMAL);
	imshow("Father - BabyFaceGenerator", FaImage / 255.0);
	resizeWindow("Father - BabyFaceGenerator", FaImage.size().width*0.5, FaImage.size().height*0.5);

	/// Hiển thị ảnh con
	namedWindow("Mother - BabyFaceGenerator", WINDOW_NORMAL);
	imshow("Mother - BabyFaceGenerator", MoImage / 255.0);
	resizeWindow("Mother - BabyFaceGenerator", FaImage.size().width*0.5, FaImage.size().height*0.5);

	Rect RectNewFace = boundingRect(Points);
	Mat ImageROI = ImgMorph(RectNewFace);

	Rect RectBabyFace = boundingRect(ChildPoint);
	Mat ImageBabyNew = ImgBaby(RectBabyFace);

	// hiện thị ảnh con khi lớn
	namedWindow("Adult - BabyFaceGenerator", WINDOW_NORMAL);
	imshow("Adult - BabyFaceGenerator", ImageROI / 255.0);
	resizeWindow("Adult - BabyFaceGenerator", ImageROI.size().width*0.75, ImageROI.size().height*0.75);
	 // hiển thị ảnh con lúc bé
	namedWindow("Baby - BabyFaceGenerator", WINDOW_NORMAL);
	imshow("Baby - BabyFaceGenerator", ImageBabyNew / 255.0);
	resizeWindow("Baby - BabyFaceGenerator", ImageBabyNew.size().width*0.75, ImageBabyNew.size().height*0.75);

	waitKey(0);
}

