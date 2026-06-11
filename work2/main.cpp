#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // ====================== 任务1：读取测试图片 ======================
    Mat img = imread("test.jpg");
    if (img.empty()) {
        cout << "❌ 错误：无法读取test.jpg，请检查文件是否在当前目录！" << endl;
        return -1;
    }
    cout << "✅ 任务1完成：成功读取测试图片" << endl << endl;

    // ====================== 任务2：输出图像基本信息 ======================
    cout << "===== 图像基本信息 =====" << endl;
    cout << "图像宽度：" << img.cols << " 像素" << endl;
    cout << "图像高度：" << img.rows << " 像素" << endl;
    cout << "通道数：" << img.channels() << endl;
    cout << "像素数据类型：" << img.type() << endl;
    cout << "总像素数：" << img.total() << endl << endl;
    cout << "✅ 任务2完成：已输出图像信息" << endl << endl;

    // ====================== 任务3：显示原图 ======================
    namedWindow("原图", WINDOW_NORMAL);
    imshow("原图", img);
    cout << "✅ 任务3完成：已显示原图" << endl << endl;

    // ====================== 任务4：转换为灰度图并显示 ======================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_NORMAL);
    imshow("灰度图", gray_img);
    cout << "✅ 任务4完成：已转换并显示灰度图" << endl << endl;

    // ====================== 任务5：保存灰度图 ======================
    imwrite("gray_output.jpg", gray_img);
    cout << "✅ 任务5完成：灰度图已保存为 gray_output.jpg" << endl << endl;

    // ====================== 任务6：像素操作 + 裁剪 ======================
    cout << "===== 任务6：像素与裁剪操作 =====" << endl;
    // 输出左上角(0,0)的BGR像素值
    Vec3b pixel = img.at<Vec3b>(0, 0);
    cout << "左上角(0,0)像素值（BGR格式）：" << endl;
    cout << "B(蓝)：" << (int)pixel[0] << endl;
    cout << "G(绿)：" << (int)pixel[1] << endl;
    cout << "R(红)：" << (int)pixel[2] << endl << endl;

    // 裁剪左上角100x100区域并保存
    Mat roi = img(Rect(0, 0, 100, 100));
    imwrite("roi.jpg", roi);
    namedWindow("裁剪区域", WINDOW_NORMAL);
    imshow("裁剪区域", roi);
    cout << "✅ 任务6完成：左上角100x100区域已保存为 roi.jpg" << endl << endl;

    // 等待按键，防止窗口闪退（必须加！）
    waitKey(0);
    destroyAllWindows();

    cout << "🎉 所有作业任务全部完成！" << endl;
    return 0;
}