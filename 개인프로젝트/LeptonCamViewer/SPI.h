/*
 * SPI testing utility (using spidev driver)
 *
 * Copyright (c) 2007  MontaVista Software, Inc.
 * Copyright (c) 2007  Anton Vorontsov <avorontsov@ru.mvista.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License.
 *
 * Cross-compile with cross-gcc -I/path/to/cross-kernel/include
 */

#ifndef SPI_H
#define SPI_H

#include <string>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>

extern int spi_cs0_fd;
extern int spi_cs1_fd;
extern unsigned char spi_mode;
extern unsigned char spi_bitsPerWord;
extern unsigned int spi_speed;

int SpiOpenPort(int spi_device, unsigned int spi_speed);
int SpiClosePort(int spi_device);


// Flip byte order of a word
#define FLIP_WORD_BYTES(word) (word >> 8) | (word << 8)
// The size of a single VoSPI packet
#define VOSPI_PACKET_BYTES 164
#define VOSPI_PACKET_SYMBOLS 160
// The maximum number of packets per segment, sufficient to include telemetry
#define VOSPI_MAX_PACKETS_PER_SEGMENT 61
// The number of packets in segments with and without telemetry lines present
#define VOSPI_PACKETS_PER_SEGMENT_NORMAL 60
#define VOSPI_PACKETS_PER_SEGMENT_TELEMETRY 61
// The number of segments per frame
#define VOSPI_SEGMENTS_PER_FRAME 4
// The maximum number of resets allowed before giving up on synchronising
#define VOSPI_MAX_SYNC_RESETS 30
// The maximum number of invalid frames before giving up and assuming we've lost sync
// FFC duration is nominally 23 frames, so we should never exceed that
#define VOSPI_MAX_INVALID_FRAMES 25

// A single VoSPI packet
typedef struct {
  uint16_t id;
  uint16_t crc;
  uint8_t symbols[VOSPI_PACKET_SYMBOLS];
} vospi_packet_t;

// A single VoSPI segment
typedef struct {
  vospi_packet_t packets[VOSPI_MAX_PACKETS_PER_SEGMENT];
  int packet_count;
} vospi_segment_t;

// A single VoSPI frame
typedef struct {
  vospi_segment_t segments[VOSPI_SEGMENTS_PER_FRAME];
} vospi_frame_t;

int vospi_init(int fd, uint32_t speed);
int sync_and_transfer_frame(int fd, vospi_frame_t* frame);
int transfer_frame(int fd, vospi_frame_t* frame);

#endif
