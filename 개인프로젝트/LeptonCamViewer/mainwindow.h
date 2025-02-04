#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QHostAddress>
#include <QNetworkInterface>
#include <QUdpSocket>
#include <time.h>

#include "iniparser.hpp"
#include "LeptonThread.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void timerEvent(QTimerEvent *event);

private:
    Ui::MainWindow *ui;
    LeptonThread *L_thread;
    int capture_timer = -1;
    int broadcast_timer = -1;
    int s_time = 0;
    int img_count = 0;
    int section_number = 1;
    time_t tm;
    bool saveImage = false;
    QString currIP;

    QUdpSocket* network_socket;

public slots:
    void setImage(QImage, int, int, int);
    void changed_thresholdValue(int);
    void precessPendingDatagrams();

    void clicked_radioBtn();
    void clicked_refreshBtn();
    void clicked_capture();
};
#endif // MAINWINDOW_H
