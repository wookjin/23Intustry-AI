#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QList<QHostAddress> addrList = QNetworkInterface::allAddresses();
    foreach(QHostAddress addr, addrList)
    {
        if(0<addr.toIPv4Address())
        {
            if(addr.toString() != "127.0.0.1")
            {
                qDebug() << addr.toString();
                currIP = addr.toString();
            }
        }
    }

    //ini file load code
    /*
     * section number
     * threshold
     *
    */

    INI::File ft;
    if(!ft.Load("./config.ini"))
    {
        qDebug() << "setting file don't exist";
    }

    int sNum, thold;
    section_number = ft.GetSection("Settings")->GetValue("section", -1).AsInt();
    thold = ft.GetSection("Settings")->GetValue("threshold", -1).AsInt();

    ui->thresholdSlider->setValue(thold);
    changed_thresholdValue(thold);

    network_socket = new QUdpSocket;
    network_socket->bind(QHostAddress(currIP),65000);
    QObject::connect(network_socket, &QUdpSocket::readyRead, this, &MainWindow::precessPendingDatagrams);

    int typeColormap = 3; // colormap_ironblack
    int typeLepton = 3; // Lepton 2.x
    int spiSpeed = 20; // SPI bus speed 20MHz
    int rangeMin = -1; //
    int rangeMax = -1; //
    int loglevel = 0;

    //create a thread to gather SPI data
    //when the thread emits updateImage, the label should update its image accordingly
    L_thread = new LeptonThread();
    L_thread->setLogLevel(loglevel);
    L_thread->useColormap(typeColormap);
    L_thread->useLepton(typeLepton);
    //spiSpeed = 3;
    qDebug()<<"SPI Speed:"<<spiSpeed;
    L_thread->useSpiSpeedMhz(spiSpeed);
    L_thread->setAutomaticScalingRange();

    if (0 <= rangeMin) L_thread->useRangeMinValue(rangeMin);
    if (0 <= rangeMax) L_thread->useRangeMaxValue(rangeMax);

    QObject::connect(ui->btn_binary, SIGNAL(clicked()), this, SLOT(clicked_radioBtn()));
    QObject::connect(ui->btn_ironblack, SIGNAL(clicked()), this, SLOT(clicked_radioBtn()));
    QObject::connect(ui->btn_grayscale, SIGNAL(clicked()), this, SLOT(clicked_radioBtn()));
    QObject::connect(ui->btn_rainbow, SIGNAL(clicked()), this, SLOT(clicked_radioBtn()));
    QObject::connect(ui->btn_refresh, SIGNAL(clicked()), this, SLOT(clicked_refreshBtn()));

    QObject::connect(ui->thresholdSlider, SIGNAL(valueChanged(int)), this, SLOT(changed_thresholdValue(int)));
    QObject::connect(ui->btn_capture, SIGNAL(released()), this, SLOT(clicked_capture()));
    QObject::connect(ui->btn_ffc, SIGNAL(released()), L_thread, SLOT(performFFC()));

    QObject::connect(L_thread, SIGNAL(updateImage(QImage, int, int, int)), this, SLOT(setImage(QImage, int, int, int)));

    ui->btn_ironblack->setChecked(true);

    L_thread->start();
    broadcast_timer = startTimer(10*1000);
}

void MainWindow::precessPendingDatagrams()
{
    QHostAddress sender;
    quint16 senderPort;

    while(network_socket->hasPendingDatagrams())
    {
        QByteArray datagram;
        datagram.resize(network_socket->pendingDatagramSize());
        network_socket->readDatagram(datagram.data(), datagram.size(), &sender, &senderPort);
        qDebug() << datagram;
    }
}

void MainWindow::clicked_refreshBtn()
{
    L_thread->refresh();
}

void MainWindow::setImage(QImage image, int minv, int avgv, int maxv)
{
    time_t t;
    t=time(NULL);

    tm = t;

    QPixmap pixmap = QPixmap::fromImage(image);
    ui->label->setFixedSize(320,240);
    int w = ui->label->width();
    int h = ui->label->height();
    ui->label->setPixmap(pixmap.scaled(w, h, Qt::KeepAspectRatio));
    ui->statusbar->showMessage(QString("Min : %1, Max : %2, Avg : %3").arg(minv).arg(maxv).arg(avgv));


    if(saveImage)
    {
        saveImage = false;
        QString fileName = QString("/home/jhh/capIMG/img_%1.jpeg").arg(QString::number(img_count));
        img_count++;
        pixmap.save(fileName);
    }
}

void MainWindow::clicked_capture()
{
    QString nSec = ui->ncut_edit->text();

    if(nSec == "")
    {
        qDebug() << "empty value";
        s_time = 1;
    }
    else
    {
        s_time = ui->ncut_edit->text().toInt();
    }

    capture_timer = startTimer(1000);
    ui->btn_capture->setEnabled(false);
}

void MainWindow::timerEvent(QTimerEvent *event)
{
    if(event->timerId() == capture_timer)
    {
        if(s_time == img_count)
        {
            killTimer(capture_timer);
            img_count=0;
            ui->btn_capture->setEnabled(true);
            return;
        }

        qDebug() << "capture timer";
        saveImage = true;
    }
    else if(event->timerId() == broadcast_timer)
    {
        time_t t;
        t=time(NULL);

        double diff = difftime(t, tm);

        // IP # SECTION # STATE # DETECTED # UPDATE TIME

        QString sendData = currIP;
        sendData.append("#");

        sendData.append(QString::number(section_number));
        sendData.append("#");

        if(diff<24*60*60)sendData.append("Normal");
        else sendData.append("NoImage");
        sendData.append("#");

        sendData.append("0"); // detect = 1, not detected = 0;

        sendData.append("#");
        sendData.append(QString::number(t));

        QByteArray dgram = sendData.toStdString().c_str();
        network_socket->writeDatagram(dgram.data(), dgram.size(), QHostAddress::Broadcast, 65000);
    }
}

void MainWindow::clicked_radioBtn()
{
    if(ui->btn_binary->isChecked())
    {
        L_thread->useColormap(4);
        qDebug() << "btn_binary->isChecked()";
    }

    if(ui->btn_grayscale->isChecked())
    {
        L_thread->useColormap(2);
        qDebug() << "btn_grayscale->isChecked()";
    }

    if(ui->btn_ironblack->isChecked())
    {

        L_thread->useColormap(3);
        qDebug() << "btn_color->isChecked()";
    }

    if(ui->btn_rainbow->isChecked())
    {
        L_thread->useColormap(1);
        qDebug() << "btn_rainbow->isChecked()";
    }

    L_thread->performFFC();
}

void MainWindow::changed_thresholdValue(int value)
{
    L_thread->changeThreshold(value);
    ui->threshold_val->setText(QString::number(value));
}

MainWindow::~MainWindow()
{
    delete ui;
}

