import tensorflow as tf
import numpy as np
import scipy.misc
import librosa
import subprocess

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

def midi_to_wav(input_path, output_path):
    subprocess.call(['timidity', input_path, '-Ow', '-o', output_path])

def midi_paths_to_numpy(input_paths, sample_rate):
    length = len(input_paths)
    waves = list()
    for i in range(length):
        midi_to_wav(input_paths[i], './'+str(i)+'.wav')
        wave, sr = librosa.load('./'+str(i)+'.wav', sample_rate, mono=False)
        waves.append(wave)
        subprocess.call(['rm', './'+str(i)+'.wav'])
    return np.transpose(np.array(waves), (0,2,1))

class TF_Logger(object):
    def __init__(self, log_dir, sample_rate=22050):
        """Create a summary writer logging to log_dir."""

        if tf.__version__.startswith('2'):
            self.writer = tf.summary.create_file_writer(log_dir)
        else:
            self.writer = tf.summary.FileWriter(log_dir)

        self.sample_rate = sample_rate

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if tf.__version__.startswith('2'):
            with tf.device("CPU:0"):
                with self.writer.as_default():
                    tf.summary.scalar(tag, value, step=step)
                    self.writer.flush()
        else:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        if tf.__version__.startswith('2'):
            with tf.device("CPU:0"):
                with self.writer.as_default():
                    # Log every image of batch
                    if len(images.shape) == 4:
                        tf.summary.image(tag, images, step, max_outputs=images.shape[0])
                    else:
                        tf.summary.image(tag, images, step)
        else:
            img_summaries = []
            for i, img in enumerate(images):
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(img).save(s, format="png")

                # Create an Image object
                img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                            height=img.shape[0],
                                            width=img.shape[1])
                # Create a Summary value
                img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

            # Create and write Summary
            summary = tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        if tf.__version__.startswith('2'):
            raise NotImplementedError
        else:
            # Create a histogram using numpy
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill the fields of the histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))

            # Drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            # Create and write Summary
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            self.writer.add_summary(summary, step)
            self.writer.flush()

    def audio_summary(self, tag, waves, step):
        # waves shape should be [batch, length, num_channels]
        """Log a list of waveform audios."""
        if tf.__version__.startswith('2'):
            with tf.device("CPU:0"):
                with self.writer.as_default():
                    # Log every wave of batch
                    if len(waves.shape) == 3:
                        tf.summary.audio(tag, waves, self.sample_rate, step, max_outputs=waves.shape[0])
                    else:
                        tf.summary.audio(tag, waves, self.sample_rate, step)
        else:
            print("Tensorboard version shoud be 2.x")
            raise NotImplementedError

    def midi_to_audio_summary(self, tag, file_paths, step):
        """Log a list of midi files to audio."""
        waves = midi_paths_to_numpy(file_paths, self.sample_rate)
        if tf.__version__.startswith('2'):
            self.audio_summary(tag, waves, step)
        else:
            print("Tensorboard version shoud be 2.x")
            raise NotImplementedError