import os
import sox
import ffmpeg
from datetime import datetime

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

raw_path='../music_data/raw'
processed_path='../music_data/preprocessed'

skip=45
sample_duration=30

track_time=datetime.now()

failed_files=[]

for base,(dirpath, dirnames, filenames) in enumerate(os.walk(raw_path)):
    if base == 0:
        genres=dirnames

for genre in genres:
    raw_genre_path=raw_path+'/'+genre
    processed_genre_path=processed_path+'/'+genre
    mkdir(processed_genre_path )
    if genre == 'Folk':
        chunks_n=4
    else:
        chunks_n=3

    for num,(dirpath, dirnames, filenames) in enumerate(os.walk(raw_genre_path)):
        for file_num, file in enumerate(filenames):
            try:
                raw_file=dirpath+'/'+file

        #        extention=sox.file_info.file_extension(raw_file)
                song_duration=sox.file_info.duration(raw_file)

                processed_file=processed_genre_path+'/'+genre+'-'+str(file_num) +'.wav' #+extention

                ffmpeg.input(raw_file).output(processed_file, ac=1).overwrite_output().run()

                n=int(song_duration-skip)//sample_duration
                if n > chunks_n: n = chunks_n

                for chunk in range(n):
                    start=chunk*sample_duration+skip

                    chunk_name=processed_file.split('.wav')[0] + '-'+ str(chunk) + '.wav' # + extention

                    tfm = sox.Transformer()
                    tfm.rate(samplerate=22050)
        #            tfm.channels(n_channels=1)
                    tfm.trim(start, start+sample_duration)
                    tfm.build_file( processed_file, chunk_name)
                os.remove(processed_file)   
            except:
                failed_files.append(file)
                continue
             
#print('Duration:',song_duration-skip)

for file in failed_files:
    print('Failed to proceed: '+file)

