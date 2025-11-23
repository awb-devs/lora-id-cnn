import numpy as np
import pmt
from gnuradio import gr
from datetime import datetime

class blk(gr.sync_block):
    """Save iq samples on triggered output"""

    def __init__(self, SF=7, N='None'):
        """arguments to this show up as params in grc"""
        gr.sync_block.__init__(
                self,
                name = 'Triggered IQ Record',
                in_sig = [np.complex64],
                out_sig = None
                )
        self.sf = SF
        self.device_note = N
        self.preamble_frame_len = 8 * int(pow(2, SF))
        # preallocated preamble array
        self.preamble_buffer = np.zeros(self.preamble_frame_len, dtype=np.complex64)
        self.collected_preamble_samples = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]

        # collect 'frame_info' from tags 
        info = None
        tags = self.get_tags_in_window(0, 0, len(in0))
        for tag in tags:
            key = pmt.to_python(tag.key)
            if key == 'frame_info':
                info = {
                        'offset': tag.offset,
                        'key': key,
                        'value': pmt.to_python(tag.value),
                        }
        
        
        if self.collected_preamble_samples == 0:
            if info is None:
                print("No Tag")
                return len(in0)
            elif not info['value']['is_header']:
                print("Not header")
                return len(in0)
            print("HEADER FOUNDDDDDDD")
            print(f"Tag at sample {info['offset']}: {info['key']} = {info['value']}")
            print(f"Frame size: {self.preamble_frame_len}")
            print(f"items read: {self.nitems_read(0)}")
            print(f"buffer len: {len(in0)}")
            start = info['offset'] - self.nitems_read(0)
            end = start + self.preamble_frame_len
        else:
            start = 0
            end = start + (self.preamble_frame_len - self.collected_preamble_samples)

        if len(in0) < end:
            end = len(in0)
            self.preamble_buffer[self.collected_preamble_samples:self.collected_preamble_samples + end - start] = in0[start:end]
            self.collected_preamble_samples += end - start
        else:
            self.preamble_buffer[self.collected_preamble_samples:self.collected_preamble_samples + end - start] = in0[start:end]
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"{self.device_note}_{timestamp}.cfile"
            self.preamble_buffer.tofile(file_name)
            self.collected_preamble_samples = 0
            print("SAVED")

        return len(in0)

