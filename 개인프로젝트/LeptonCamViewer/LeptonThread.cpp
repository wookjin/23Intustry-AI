#include <iostream>
#include <QDebug>

#include "LeptonThread.h"

#include "Palettes.h"
#include "SPI.h"
#include "Lepton_I2C.h"

#define PACKET_SIZE 164
#define PACKET_SIZE_UINT16 (PACKET_SIZE/2)
#define PACKETS_PER_FRAME 60
#define FRAME_SIZE_UINT16 (PACKET_SIZE_UINT16*PACKETS_PER_FRAME)
#define FPS 27;

#define FRAME_BUF_SIZE 8
vospi_frame_t* frame_buf[FRAME_BUF_SIZE];
// Positions of the reader and writer in the frame buffer
int reader = 0, writer = 0;



LeptonThread::LeptonThread() : QThread()
{
	//
	loglevel = 0;

	//
	typeColormap = 3; // 1:colormap_rainbow  /  2:colormap_grayscale  /  3:colormap_ironblack(default)
	selectedColormap = colormap_ironblack;
	selectedColormapSize = get_size_colormap_ironblack();

	//
	typeLepton = 2; // 2:Lepton 2.x  / 3:Lepton 3.x
	myImageWidth = 80;
	myImageHeight = 60;

	//
	spiSpeed = 20 * 1000 * 1000; // SPI bus speed 20MHz

	// min/max value for scaling
	autoRangeMin = true;
	autoRangeMax = true;
	rangeMin = 30000;
	rangeMax = 32000;
}

LeptonThread::~LeptonThread() {
}

void LeptonThread::setLogLevel(uint16_t newLoglevel)
{
	loglevel = newLoglevel;
}

void LeptonThread::useColormap(int newTypeColormap)
{
	switch (newTypeColormap) {
	case 1:
		typeColormap = 1;
        selectedColormap = colormap_rainbow;
		selectedColormapSize = get_size_colormap_rainbow();
		break;
	case 2:
		typeColormap = 2;
		selectedColormap = colormap_grayscale;
		selectedColormapSize = get_size_colormap_grayscale();
		break;
    case 3:
		typeColormap = 3;
		selectedColormap = colormap_ironblack;
		selectedColormapSize = get_size_colormap_ironblack();
		break;
    case 4:
        typeColormap = 4;
        selectedColormap = colormap_grayscale;
        selectedColormapSize = get_size_colormap_grayscale();
	}
}

void LeptonThread::useLepton(int newTypeLepton)
{
	switch (newTypeLepton) {
	case 3:
		typeLepton = 3;
		myImageWidth = 160;
		myImageHeight = 120;
		break;
	default:
		typeLepton = 2;
		myImageWidth = 80;
		myImageHeight = 60;
	}
}

void LeptonThread::useSpiSpeedMhz(unsigned int newSpiSpeed)
{
	spiSpeed = newSpiSpeed * 1000 * 1000;
}

void LeptonThread::setAutomaticScalingRange()
{
	autoRangeMin = true;
	autoRangeMax = true;
}

void LeptonThread::useRangeMinValue(uint16_t newMinValue)
{
	autoRangeMin = false;
	rangeMin = newMinValue;
}

void LeptonThread::useRangeMaxValue(uint16_t newMaxValue)
{
	autoRangeMax = false;
	rangeMax = newMaxValue;
}




void LeptonThread::run()
{
	myImage = QImage(myImageWidth, myImageHeight, QImage::Format_RGB888);

    //const int *colormap = selectedColormap;
    //const int colormapSize = selectedColormapSize;
	uint16_t minValue = rangeMin;
	uint16_t maxValue = rangeMax;
	float diff = maxValue - minValue;
	float scale = 255/diff;
	uint16_t n_wrong_segment = 0;
	uint16_t n_zero_value_drop_frame = 0;

    SpiOpenPort(0, spiSpeed);
    cv::Mat tempMat8 = cv::Mat::zeros(120, 160, CV_8UC1);
    while(true)
    {
        int resets = 0;
        int segmentNumber = -1;
        for(int j=0;j<PACKETS_PER_FRAME;j++) {

            read(spi_cs0_fd, result+sizeof(uint8_t)*PACKET_SIZE*j, sizeof(uint8_t)*PACKET_SIZE);
            int packetNumber = result[j*PACKET_SIZE+1];

            if(packetNumber != j) {
                j = -1;
                resets += 1;
                usleep(1000);

                if(resets == 750)
                {
                    SpiClosePort(0);
                    lepton_reboot();
                    n_wrong_segment = 0;
                    n_zero_value_drop_frame = 0;
                    usleep(750000);
                    SpiOpenPort(0, spiSpeed);
                }

                continue;
            }

            if ((typeLepton == 3) && (packetNumber == 20))
            {
                segmentNumber = (result[j*PACKET_SIZE] >> 4) & 0x0f;
                if ((segmentNumber < 1) || (4 < segmentNumber)) {
                    //log_message(10, "[ERROR] Wrong segment number " + std::to_string(segmentNumber));
                    //qDebug()<<"[ERROR] Wrong segment number " << segmentNumber;

                    break;
                }
            }
        }

        if(resets >= 30) {
            log_message(3, "done reading, resets: " + std::to_string(resets));
            qDebug()<<"done reading, resets: " << resets;
        }

        int iSegmentStart = 1;
        int iSegmentStop;
        if (typeLepton == 3) {
            if ((segmentNumber < 1) || (4 < segmentNumber)) {
                n_wrong_segment++;
                if ((n_wrong_segment % 12) == 0) {
                    log_message(5, "[WARNING] Got wrong segment number continuously " + std::to_string(n_wrong_segment) + " times");
                }
                continue;
            }
            if (n_wrong_segment != 0) {
                log_message(8, "[WARNING] Got wrong segment number continuously " + std::to_string(n_wrong_segment) + " times [RECOVERED] : " + std::to_string(segmentNumber));
                n_wrong_segment = 0;
            }

            memcpy(shelf[segmentNumber - 1], result, sizeof(uint8_t) * PACKET_SIZE*PACKETS_PER_FRAME);
            if (segmentNumber != 4) {
                continue;
            }
            iSegmentStop = 4;
        }
        else {
            memcpy(shelf[0], result, sizeof(uint8_t) * PACKET_SIZE*PACKETS_PER_FRAME);
            iSegmentStop = 1;
        }

        if ((autoRangeMin == true) || (autoRangeMax == true)) {
            if (autoRangeMin == true) {
                maxValue = 65535;
            }
            if (autoRangeMax == true) {
                maxValue = 0;
            }
            for(int iSegment = iSegmentStart; iSegment <= iSegmentStop; iSegment++) {
                for(int i=0;i<FRAME_SIZE_UINT16;i++)
                {

                    if(i % PACKET_SIZE_UINT16 < 2) {
                        continue;
                    }

                    uint16_t value = (shelf[iSegment - 1][i*2] << 8) + shelf[iSegment - 1][i*2+1];
                    if (value == 0)
                    {
                        continue;
                    }
                    if ((autoRangeMax == true) && (value > maxValue)) {
                        maxValue = value;
                    }
                    if ((autoRangeMin == true) && (value < minValue)) {
                        minValue = value;
                    }
                }
            }
            diff = maxValue - minValue;
            scale = 255/diff;
        }

        int row, column;
        uint16_t value;
        uint16_t valueFrameBuffer;
        QRgb color;

        int minv = 100;
        int maxv = 0;
        int avgv = 0;

        for(int iSegment = iSegmentStart; iSegment <= iSegmentStop; iSegment++) {
            int ofsRow = 30 * (iSegment - 1);
            for(int i=0;i<FRAME_SIZE_UINT16;i++)
            {
                if(i % PACKET_SIZE_UINT16 < 2) {
                    continue;
                }

                valueFrameBuffer = (shelf[iSegment - 1][i*2] << 8) + shelf[iSegment - 1][i*2+1];
                if (valueFrameBuffer == 0)
                {

                    n_zero_value_drop_frame++;
                    if ((n_zero_value_drop_frame % 12) == 0) {
                        log_message(5, "[WARNING] Found zero-value. Drop the frame continuously " + std::to_string(n_zero_value_drop_frame) + " times");
                    }
                    break;
                }

                if(typeColormap!=4)
                {
                    value = (valueFrameBuffer - minValue) * scale;
                }
                else
                {
                    value = (valueFrameBuffer/100.0) - 273.5;

                    if(value < minv) minv = value;
                    if(value > maxv) maxv = value;

                    avgv += value;
                }

                int ofs_r = 3 * value + 0; if (selectedColormapSize <= ofs_r) ofs_r = selectedColormapSize - 1;
                int ofs_g = 3 * value + 1; if (selectedColormapSize <= ofs_g) ofs_g = selectedColormapSize - 1;
                int ofs_b = 3 * value + 2; if (selectedColormapSize <= ofs_b) ofs_b = selectedColormapSize - 1;
                color = qRgb(selectedColormap[ofs_r], selectedColormap[ofs_g], selectedColormap[ofs_b]);
                if (typeLepton == 3) {
                    column = (i % PACKET_SIZE_UINT16) - 2 + (myImageWidth / 2) * ((i % (PACKET_SIZE_UINT16 * 2)) / PACKET_SIZE_UINT16);
                    row = i / PACKET_SIZE_UINT16 / 2 + ofsRow;
                }
                else {
                    column = (i % PACKET_SIZE_UINT16) - 2;
                    row = i / PACKET_SIZE_UINT16;
                }
                myImage.setPixel(column, row, color);
            }
        }

        if (n_zero_value_drop_frame != 0) {
            log_message(8, "[WARNING] Found zero-value. Drop the frame continuously " + std::to_string(n_zero_value_drop_frame) + " times [RECOVERED]");
            n_zero_value_drop_frame = 0;
        }

        if(typeColormap==4)
        {
            tempMat8 = QImageToMat(myImage);

            cv::Mat norm_img, norm_img_color;
            cv::normalize(tempMat8, norm_img, 0, 255, cv::NORM_MINMAX);
            cv::applyColorMap(norm_img, norm_img_color, cv::COLORMAP_JET);

            cv::Mat binary;
            cv::threshold(tempMat8, binary, change_Threshold, 255, cv::THRESH_BINARY);

            QImage qimg = matToQImage(binary);
            avgv = avgv/(row*column);
            emit updateImage(qimg, minv, avgv, maxv);
        }
        else
        {
            emit updateImage(myImage, 0, 0, 0);
        }
    }
	
	SpiClosePort(0);
}

void LeptonThread::changeThreshold(int val)
{
    change_Threshold = val;
}

void LeptonThread::refresh()
{
    string path("/home/jhh/capIMG/*.jpeg");
    vector<string> str;

    glob(path, str, false);

    Mat result;

    //QString temp = QString("img_%1.jpeg").arg(0);
    //cv::Mat sumimg = cv::imread("img_%1.jpeg", 0);
    for(int i=0;i<str.size();i++)
    {
        QString temp = QString("img_%1.jpeg").arg(i);
        cv::Mat sumimg = cv::imread(str[i], cv::IMREAD_GRAYSCALE);
//cv::Mat sumimg = cv::imread(temp.toLocal8Bit().data(), cv::IMREAD_GRAYSCALE);
        if(!sumimg.empty())
        {
            cv::hconcat(sumimg, result);

        }
    }

    if(!result.empty())
        cv::imshow("Merged Image", result);

}

void LeptonThread::performFFC()
{
	lepton_perform_ffc();
}

void LeptonThread::log_message(uint16_t level, std::string msg)
{
	if (level <= loglevel) {
		std::cerr << msg << std::endl;
	}
}

