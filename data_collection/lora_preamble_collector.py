#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: lora_preamble_collector
# Author: awb-devs
# Description: This flow automatically records the preamble at Nyquist minimum sampling rate for incoming lora signals
# GNU Radio version: 3.10.11.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
import gnuradio.lora_sdr as lora_sdr
import lora_preamble_collector_epy_block_0 as epy_block_0  # embedded python block
import numpy as np
import sip
import threading



class lora_preamble_collector(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "lora_preamble_collector", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("lora_preamble_collector")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "lora_preamble_collector")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2000000
        self.lora_62bandwidth = lora_62bandwidth = 62500
        self.lora_250bandwidth = lora_250bandwidth = 250000
        self.lora_125bandwidth = lora_125bandwidth = 125000
        self.sync_word = sync_word = [0x12]
        self.soft_decoding = soft_decoding = True
        self.preamble_length = preamble_length = 8
        self.payload_length = payload_length = 237
        self.impl_head = impl_head = False
        self.has_crc = has_crc = True
        self.cr_48 = cr_48 = 8
        self.cr_47 = cr_47 = 3
        self.cr_46 = cr_46 = 2
        self.cr_45 = cr_45 = 1
        self.cr_44 = cr_44 = 0
        self.center_62KHz = center_62KHz = 916218750
        self.center_250KHz = center_250KHz = 906875000
        self.center_125KHz = center_125KHz = 904437500
        self.bandpass62k = bandpass62k = firdes.complex_band_pass(1.0, samp_rate, -lora_62bandwidth/2, lora_62bandwidth/2, lora_62bandwidth/10, window.WIN_HAMMING, 6.76)
        self.bandpass250k = bandpass250k = firdes.complex_band_pass(1.0, samp_rate, -lora_250bandwidth/2, lora_250bandwidth/2, lora_250bandwidth/10, window.WIN_HAMMING, 6.76)
        self.bandpass125k = bandpass125k = firdes.complex_band_pass(1.0, samp_rate, -lora_125bandwidth/2, lora_125bandwidth/2, lora_125bandwidth/10, window.WIN_HAMMING, 6.76)

        ##################################################
        # Blocks
        ##################################################

        self.soapy_hackrf_source_0 = None
        dev = 'driver=hackrf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_hackrf_source_0 = soapy.source(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)
        self.soapy_hackrf_source_0.set_sample_rate(0, samp_rate)
        self.soapy_hackrf_source_0.set_bandwidth(0, samp_rate)
        self.soapy_hackrf_source_0.set_frequency(0, center_250KHz)
        self.soapy_hackrf_source_0.set_gain(0, 'AMP', False)
        self.soapy_hackrf_source_0.set_gain(0, 'LNA', min(max(16, 0.0), 40.0))
        self.soapy_hackrf_source_0.set_gain(0, 'VGA', min(max(16, 0.0), 62.0))
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=5,
                taps=[],
                fractional_bw=0)
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            8192, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            center_250KHz, #fc
            250000, #bw
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.qwidget(), Qt.QWidget)

        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_win)
        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(impl_head, cr_45, payload_length, has_crc, 2, True)
        self.lora_sdr_hamming_dec_0 = lora_sdr.hamming_dec(soft_decoding)
        self.lora_sdr_gray_mapping_0 = lora_sdr.gray_mapping( soft_decoding)
        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(center_250KHz, lora_250bandwidth, 7, impl_head, sync_word, 4,preamble_length)
        self.lora_sdr_fft_demod_0 = lora_sdr.fft_demod( soft_decoding, True)
        self.lora_sdr_dewhitening_0 = lora_sdr.dewhitening()
        self.lora_sdr_deinterleaver_0 = lora_sdr.deinterleaver( soft_decoding)
        self.lora_sdr_crc_verif_0 = lora_sdr.crc_verif( 2, False)
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccc((int(samp_rate/(lora_250bandwidth * 4))), bandpass250k, 0, samp_rate)
        self.freq_xlating_fir_filter_xxx_0.set_min_output_buffer(17000)
        self.epy_block_0 = epy_block_0.blk(SF=7, N='iq_samples/test/test')
        self.blocks_throttle2_0 = blocks.throttle( gr.sizeof_gr_complex*1, samp_rate, True, 0 if "auto" == "auto" else max( int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1) )
        self.blocks_correctiq_0 = blocks.correctiq()


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.lora_sdr_header_decoder_0, 'frame_info'), (self.lora_sdr_frame_sync_0, 'frame_info'))
        self.connect((self.blocks_correctiq_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.blocks_correctiq_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.lora_sdr_frame_sync_0, 0))
        self.connect((self.lora_sdr_deinterleaver_0, 0), (self.lora_sdr_hamming_dec_0, 0))
        self.connect((self.lora_sdr_dewhitening_0, 0), (self.lora_sdr_crc_verif_0, 0))
        self.connect((self.lora_sdr_fft_demod_0, 0), (self.lora_sdr_gray_mapping_0, 0))
        self.connect((self.lora_sdr_frame_sync_0, 0), (self.epy_block_0, 0))
        self.connect((self.lora_sdr_frame_sync_0, 0), (self.lora_sdr_fft_demod_0, 0))
        self.connect((self.lora_sdr_gray_mapping_0, 0), (self.lora_sdr_deinterleaver_0, 0))
        self.connect((self.lora_sdr_hamming_dec_0, 0), (self.lora_sdr_header_decoder_0, 0))
        self.connect((self.lora_sdr_header_decoder_0, 0), (self.lora_sdr_dewhitening_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.qtgui_waterfall_sink_x_0, 0))
        self.connect((self.soapy_hackrf_source_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.soapy_hackrf_source_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "lora_preamble_collector")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_bandpass125k(firdes.complex_band_pass(1.0, self.samp_rate, -self.lora_125bandwidth/2, self.lora_125bandwidth/2, self.lora_125bandwidth/10, window.WIN_HAMMING, 6.76))
        self.set_bandpass250k(firdes.complex_band_pass(1.0, self.samp_rate, -self.lora_250bandwidth/2, self.lora_250bandwidth/2, self.lora_250bandwidth/10, window.WIN_HAMMING, 6.76))
        self.set_bandpass62k(firdes.complex_band_pass(1.0, self.samp_rate, -self.lora_62bandwidth/2, self.lora_62bandwidth/2, self.lora_62bandwidth/10, window.WIN_HAMMING, 6.76))
        self.blocks_throttle2_0.set_sample_rate(self.samp_rate)
        self.soapy_hackrf_source_0.set_sample_rate(0, self.samp_rate)
        self.soapy_hackrf_source_0.set_bandwidth(0, self.samp_rate)

    def get_lora_62bandwidth(self):
        return self.lora_62bandwidth

    def set_lora_62bandwidth(self, lora_62bandwidth):
        self.lora_62bandwidth = lora_62bandwidth
        self.set_bandpass62k(firdes.complex_band_pass(1.0, self.samp_rate, -self.lora_62bandwidth/2, self.lora_62bandwidth/2, self.lora_62bandwidth/10, window.WIN_HAMMING, 6.76))

    def get_lora_250bandwidth(self):
        return self.lora_250bandwidth

    def set_lora_250bandwidth(self, lora_250bandwidth):
        self.lora_250bandwidth = lora_250bandwidth
        self.set_bandpass250k(firdes.complex_band_pass(1.0, self.samp_rate, -self.lora_250bandwidth/2, self.lora_250bandwidth/2, self.lora_250bandwidth/10, window.WIN_HAMMING, 6.76))

    def get_lora_125bandwidth(self):
        return self.lora_125bandwidth

    def set_lora_125bandwidth(self, lora_125bandwidth):
        self.lora_125bandwidth = lora_125bandwidth
        self.set_bandpass125k(firdes.complex_band_pass(1.0, self.samp_rate, -self.lora_125bandwidth/2, self.lora_125bandwidth/2, self.lora_125bandwidth/10, window.WIN_HAMMING, 6.76))

    def get_sync_word(self):
        return self.sync_word

    def set_sync_word(self, sync_word):
        self.sync_word = sync_word

    def get_soft_decoding(self):
        return self.soft_decoding

    def set_soft_decoding(self, soft_decoding):
        self.soft_decoding = soft_decoding

    def get_preamble_length(self):
        return self.preamble_length

    def set_preamble_length(self, preamble_length):
        self.preamble_length = preamble_length

    def get_payload_length(self):
        return self.payload_length

    def set_payload_length(self, payload_length):
        self.payload_length = payload_length

    def get_impl_head(self):
        return self.impl_head

    def set_impl_head(self, impl_head):
        self.impl_head = impl_head

    def get_has_crc(self):
        return self.has_crc

    def set_has_crc(self, has_crc):
        self.has_crc = has_crc

    def get_cr_48(self):
        return self.cr_48

    def set_cr_48(self, cr_48):
        self.cr_48 = cr_48

    def get_cr_47(self):
        return self.cr_47

    def set_cr_47(self, cr_47):
        self.cr_47 = cr_47

    def get_cr_46(self):
        return self.cr_46

    def set_cr_46(self, cr_46):
        self.cr_46 = cr_46

    def get_cr_45(self):
        return self.cr_45

    def set_cr_45(self, cr_45):
        self.cr_45 = cr_45

    def get_cr_44(self):
        return self.cr_44

    def set_cr_44(self, cr_44):
        self.cr_44 = cr_44

    def get_center_62KHz(self):
        return self.center_62KHz

    def set_center_62KHz(self, center_62KHz):
        self.center_62KHz = center_62KHz

    def get_center_250KHz(self):
        return self.center_250KHz

    def set_center_250KHz(self, center_250KHz):
        self.center_250KHz = center_250KHz
        self.qtgui_waterfall_sink_x_0.set_frequency_range(self.center_250KHz, 250000)
        self.soapy_hackrf_source_0.set_frequency(0, self.center_250KHz)

    def get_center_125KHz(self):
        return self.center_125KHz

    def set_center_125KHz(self, center_125KHz):
        self.center_125KHz = center_125KHz

    def get_bandpass62k(self):
        return self.bandpass62k

    def set_bandpass62k(self, bandpass62k):
        self.bandpass62k = bandpass62k

    def get_bandpass250k(self):
        return self.bandpass250k

    def set_bandpass250k(self, bandpass250k):
        self.bandpass250k = bandpass250k
        self.freq_xlating_fir_filter_xxx_0.set_taps(self.bandpass250k)

    def get_bandpass125k(self):
        return self.bandpass125k

    def set_bandpass125k(self, bandpass125k):
        self.bandpass125k = bandpass125k




def main(top_block_cls=lora_preamble_collector, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
