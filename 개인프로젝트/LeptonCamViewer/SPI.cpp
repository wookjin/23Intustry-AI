#include "SPI.h"

#include <QDebug>

int spi_cs0_fd = -1;
int spi_cs1_fd = -1;

unsigned char spi_mode = SPI_MODE_3;
unsigned char spi_bitsPerWord = 8;
unsigned int spi_speed = 10000000;

int SpiOpenPort (int spi_device, unsigned int useSpiSpeed)
{
	int status_value = -1;
	int *spi_cs_fd;


	//----- SET SPI MODE -----
	//SPI_MODE_0 (0,0)  CPOL=0 (Clock Idle low level), CPHA=0 (SDO transmit/change edge active to idle)
	//SPI_MODE_1 (0,1)  CPOL=0 (Clock Idle low level), CPHA=1 (SDO transmit/change edge idle to active)
	//SPI_MODE_2 (1,0)  CPOL=1 (Clock Idle high level), CPHA=0 (SDO transmit/change edge active to idle)
	//SPI_MODE_3 (1,1)  CPOL=1 (Clock Idle high level), CPHA=1 (SDO transmit/change edge idle to active)
	spi_mode = SPI_MODE_3;

	//----- SET BITS PER WORD -----
	spi_bitsPerWord = 8;

	//----- SET SPI BUS SPEED -----
	spi_speed = useSpiSpeed;				//1000000 = 1MHz (1uS per bit)

	if (spi_device)
		spi_cs_fd = &spi_cs1_fd;
	else
		spi_cs_fd = &spi_cs0_fd;


	if (spi_device)
		*spi_cs_fd = open(std::string("/dev/spidev0.1").c_str(), O_RDWR);
	else
		*spi_cs_fd = open(std::string("/dev/spidev0.0").c_str(), O_RDWR);


	if (*spi_cs_fd < 0)
	{
		perror("Error - Could not open SPI device");        
        qDebug()<<"Error - Could not open SPI device";
		exit(1);
	}

	status_value = ioctl(*spi_cs_fd, SPI_IOC_WR_MODE, &spi_mode);
	if(status_value < 0)
	{
		perror("Could not set SPIMode (WR)...ioctl fail");
        qDebug()<<"Could not set SPIMode (WR)...ioctl fail";
		exit(1);
	}

	status_value = ioctl(*spi_cs_fd, SPI_IOC_RD_MODE, &spi_mode);
	if(status_value < 0)
	{
		perror("Could not set SPIMode (RD)...ioctl fail");
        qDebug()<<"Could not set SPIMode (RD)...ioctl fail";
		exit(1);
	}

	status_value = ioctl(*spi_cs_fd, SPI_IOC_WR_BITS_PER_WORD, &spi_bitsPerWord);
	if(status_value < 0)
	{
		perror("Could not set SPI bitsPerWord (WR)...ioctl fail");
        qDebug()<<"Could not set SPI bitsPerWord (WR)...ioctl fail";
		exit(1);
	}

	status_value = ioctl(*spi_cs_fd, SPI_IOC_RD_BITS_PER_WORD, &spi_bitsPerWord);
	if(status_value < 0)
	{
		perror("Could not set SPI bitsPerWord(RD)...ioctl fail");
        qDebug()<<"Could not set SPI bitsPerWord(RD)...ioctl fail";
		exit(1);
	}

	status_value = ioctl(*spi_cs_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed);
	if(status_value < 0)
	{
		perror("Could not set SPI speed (WR)...ioctl fail");
        qDebug()<<"Could not set SPI speed (WR)...ioctl fail";
		exit(1);
	}

	status_value = ioctl(*spi_cs_fd, SPI_IOC_RD_MAX_SPEED_HZ, &spi_speed);
	if(status_value < 0)
	{
		perror("Could not set SPI speed (RD)...ioctl fail");
        qDebug()<<"Could not set SPI speed (RD)...ioctl fail";
		exit(1);
	}


    qDebug()<<"SPI Open"<<status_value;
	return(status_value);
}

int SpiClosePort(int spi_device)
{
		int status_value = -1;
	int *spi_cs_fd;

	if (spi_device)
		spi_cs_fd = &spi_cs1_fd;
	else
		spi_cs_fd = &spi_cs0_fd;


	status_value = close(*spi_cs_fd);
	if(status_value < 0)
	{
		perror("Error - Could not close SPI device");        
        qDebug()<<"Error - Could not close SPI device";
		exit(1);
	}
    qDebug()<<"SPI Close Port"<<status_value;
	return(status_value);
}

/**
 * Initialise the VoSPI interface.
 */
int vospi_init(int fd, uint32_t speed)
{
  // Set the various SPI parameters
  qDebug()<<"setting SPI device mode...";
  uint16_t mode = SPI_MODE_3;
  if (ioctl(fd, SPI_IOC_WR_MODE, &mode) == -1) {
    qDebug()<<"SPI: failed to set mode";
    return -1;
  }

  qDebug()<<"setting SPI bits/word...";
  uint8_t bits = 8;
  if (ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits) == -1) {
    qDebug()<<"SPI: failed to set the bits per word option";
    return -1;
  }

  qDebug()<<"setting SPI max clock speed...";
  if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) == -1) {
    qDebug()<<"SPI: failed to set the max speed option";
    return -1;
  }
  return 1;
}

/**
 * Transfer a single VoSPI segment.
 * Returns the number of successfully-transferred segments (0 or 1).
 */
int transfer_segment(int fd, vospi_segment_t* segment)
{
  // Perform the spidev transfer
  if (read(fd, &segment->packets[0], VOSPI_PACKET_BYTES) < 1) {
    qDebug()<<"SPI: failed to transfer packet";
    return 0;
  }

  // Flip the byte order of the ID & CRC
  segment->packets[0].id = FLIP_WORD_BYTES(segment->packets[0].id);
  segment->packets[0].crc = FLIP_WORD_BYTES(segment->packets[0].crc);

  qDebug()<< segment->packets[0].id<<" ////// "<< segment->packets[0].crc;

  while ((segment->packets[0].id & 0x0f00) == 0x0f00) {
    // It was a discard packet, try receiving another packet into the same buf
    read(fd, &segment->packets[0], VOSPI_PACKET_BYTES);
  }

  // Read the remaining packets
  if (read(fd, &segment->packets[1], VOSPI_PACKET_BYTES * (segment->packet_count - 1)) < 1) {
    qDebug()<<
            "SPI: failed to transfer the rest of the segment - "
            <<
            "Check to ensure that the bufsiz module parameter for spidev is set to > %d bytes"
            << VOSPI_PACKET_BYTES * segment->packet_count ;
    return 0;
  }

  // Flip the byte order for the rest of the packet IDs
  for (int i = 1; i < segment->packet_count; i ++) {
    segment->packets[i].id = FLIP_WORD_BYTES(segment->packets[i].id);
    segment->packets[i].crc = FLIP_WORD_BYTES(segment->packets[i].crc);
  }

  return 1;
}
/**
 * Synchroise the VoSPI stream and transfer a single frame.
 * Returns the number of successfully-transferred frames (0 or 1).
 */
int sync_and_transfer_frame(int fd, vospi_frame_t* frame)
{
  // Keep streaming segments until we receive a valid, first segment to sync
  qDebug()<<"synchronising with first segment";
  uint16_t packet_20_num;
  uint8_t ttt_bits, resets = 0;

  while (1) {

      // Stream a first segment
      qDebug()<<"receiving first segment...";
      if (!transfer_segment(fd, &frame->segments[0])) {
       qDebug()<<"failed to receive the first segment";
        return 0;
      }

      // If the packet number isn't even correct, we'll reset the bus to sync
      packet_20_num = frame->segments[0].packets[20].id & 0xff;
      if (packet_20_num != 20) {
          // Deselect the chip, wait 200ms with CS deasserted
          qDebug()<<"packet 20 ID was %d - deasserting CS & waiting to reset..."<< packet_20_num;
          usleep(185000);

          if (++resets >= VOSPI_MAX_SYNC_RESETS) {
              qDebug()<<"too many resets while synchronising :", resets;
              return 0;
          }

          continue;
      }

      // Check we're looking at the first segment, if not, just keep reading until we get there
      ttt_bits = frame->segments[0].packets[20].id >> 12;
      qDebug()<<"TTT bits were: %dm P20 Num: "<< ttt_bits, packet_20_num;
      if (ttt_bits == 1) {
        break;
      }
  }

  // Receive the remaining segments
  for (int seg = 1; seg < VOSPI_SEGMENTS_PER_FRAME; seg ++) {
    transfer_segment(fd, &frame->segments[seg]);
  }

  return 1;
}

/**
 * Transfer a frame.
 * Assumes that we're already synchronised with the VoSPI stream.
 */
int transfer_frame(int fd, vospi_frame_t* frame)
{
  uint8_t ttt_bits, restarts = 0;

  // Receive all segments
  for (int seg = 0; seg < VOSPI_SEGMENTS_PER_FRAME; seg ++) {
    transfer_segment(fd, &frame->segments[seg]);

    ttt_bits = frame->segments[seg].packets[20].id >> 12;
    if (ttt_bits != seg + 1) {
      seg --;
      if (restarts ++ > VOSPI_MAX_INVALID_FRAMES * 4) {
        qDebug()<<"too many invalid frames - need to resync";
        return 0;
      }
      continue;
    }
  }

  return 1;
}
