import os
import shutil
from pydub import AudioSegment

data_dir = "download"
proc_dir = "raw"

# Heart Auscultation Locations
pv_prefix = "PV"
av_prefix = "AV"
tv_prefix = "TV"
mv_prefix = "MV"

murmur_suffix = "MR"
healthy_suffix = "OK"

def main():
	shutil.rmtree("./" + proc_dir)

	if not os.path.exists("./" + proc_dir + "/" + pv_prefix + "_" + murmur_suffix):
		os.makedirs("./" + proc_dir + "/" + pv_prefix + "_" + murmur_suffix)
	if not os.path.exists("./" + proc_dir + "/" + av_prefix + "_" + murmur_suffix):
		os.makedirs("./" + proc_dir + "/" + av_prefix + "_" + murmur_suffix)
	if not os.path.exists("./" + proc_dir + "/" + tv_prefix + "_" + murmur_suffix):
		os.makedirs("./" + proc_dir + "/" + tv_prefix + "_" + murmur_suffix)
	if not os.path.exists("./" + proc_dir + "/" + mv_prefix + "_" + murmur_suffix):
		os.makedirs("./" + proc_dir + "/" + mv_prefix + "_" + murmur_suffix)
	if not os.path.exists("./" + proc_dir + "/" + pv_prefix + "_" + healthy_suffix):
		os.makedirs("./" + proc_dir + "/" + pv_prefix + "_" + healthy_suffix)
	if not os.path.exists("./" + proc_dir + "/" + av_prefix + "_" + healthy_suffix):
		os.makedirs("./" + proc_dir + "/" + av_prefix + "_" + healthy_suffix)
	if not os.path.exists("./" + proc_dir + "/" + tv_prefix + "_" + healthy_suffix):
		os.makedirs("./" + proc_dir + "/" + tv_prefix + "_" + healthy_suffix)
	if not os.path.exists("./" + proc_dir + "/" + mv_prefix + "_" + healthy_suffix):
		os.makedirs("./" + proc_dir + "/" + mv_prefix + "_" + healthy_suffix)

	for file_name in os.listdir("./" + data_dir):
		if file_name.endswith(".txt"):
			with open(os.path.join("./" + data_dir, file_name)) as file:
				cnt = file.readlines()
				file.close()
				
				pData = {
					"pID": None,
					"pv": {
						"wav": None,
						"tsv": None,
						"mr": False,
					},
					"av": {
						"wav": None,
						"tsv": None,
						"mr": False,
					},
					"tv": {
						"wav": None,
						"tsv": None,
						"mr": False,
					},
					"mv": {
						"wav": None,
						"tsv": None,
						"mr": False,
					}
				}
				
				isStartLine = True
				for line in cnt:
					if line[-1] == '\n':
						line = line[:-1]
					texts = line.split(" ")
					if isStartLine:
						pData["pID"] = int(texts[0])
						isStartLine = False
					else:
						if texts[0] == pv_prefix:
							for text in texts:
								if text[-3:] == "wav":
									pData["pv"]["wav"] = text
								if text[-3:] == "tsv":
									pData["pv"]["tsv"] = text
						if texts[0] == av_prefix:
							for text in texts:
								if text[-3:] == "wav":
									pData["av"]["wav"] = text
								if text[-3:] == "tsv":
									pData["av"]["tsv"] = text
						if texts[0] == tv_prefix:
							for text in texts:
								if text[-3:] == "wav":
									pData["tv"]["wav"] = text
								if text[-3:] == "tsv":
									pData["tv"]["tsv"] = text
						if texts[0] == mv_prefix:
							for text in texts:
								if text[-3:] == "wav":
									pData["mv"]["wav"] = text
								if text[-3:] == "tsv":
									pData["mv"]["tsv"] = text
						if texts[0] == "#Murmur":
							murmur_locs = texts[2].split("+")
							for murmur_loc in murmur_locs:
								if murmur_loc == pv_prefix:
									pData["pv"]["mr"] = True
								if murmur_loc == av_prefix:
									pData["av"]["mr"] = True
								if murmur_loc == tv_prefix:
									pData["tv"]["mr"] = True
								if murmur_loc == mv_prefix:
									pData["mv"]["mr"] = True
									
				if pData["pv"]["wav"] is not None and pData["pv"]["tsv"] is not None:
					with open(os.path.join("./" + data_dir, pData["pv"]["tsv"])) as tsvfile:
						tsvcnt = tsvfile.readlines()
						tsvfile.close()
						
						record = AudioSegment.from_wav(os.path.join("./" + data_dir, pData["pv"]["wav"]))
						
						isStartBeat = False
						beatIndex = 0
						soundIndex = 0
						startMS = None
						endMS = None
						for line in tsvcnt:
							if line[-1] == '\n':
								line = line[:-1]
							tsvTabs = line.split('\t')
							
							if isStartBeat == False:
								if int(tsvTabs[2]) == 1:
									isStartBeat = True
									beatIndex = 1
									startMS = int(float(tsvTabs[0]) * 1000)
							else:
								if int(tsvTabs[2]) == beatIndex + 1:
									beatIndex = beatIndex + 1
									if beatIndex == 4:
										endMS = int(float(tsvTabs[1]) * 1000)
										beatIndex = 0
										
										# Splice and store heartbeat as WAV
										heartsound = record[startMS:endMS]
										if pData["pv"]["mr"] == True:
											output_filename = os.path.join("./" + proc_dir + "/" + pv_prefix + "_" + murmur_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										else:
											output_filename = os.path.join("./" + proc_dir + "/" + pv_prefix + "_" + healthy_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										heartsound.export(output_filename, format="wav")
										
										soundIndex = soundIndex + 1
										
									if beatIndex == 1:
										startMS = int(float(tsvTabs[0]) * 1000)
								else:
									beatIndex = 0
									
				if pData["av"]["wav"] is not None and pData["av"]["tsv"] is not None:
					with open(os.path.join("./" + data_dir, pData["av"]["tsv"])) as tsvfile:
						tsvcnt = tsvfile.readlines()
						tsvfile.close()
						
						record = AudioSegment.from_wav(os.path.join("./" + data_dir, pData["av"]["wav"]))
						
						isStartBeat = False
						beatIndex = 0
						soundIndex = 0
						startMS = None
						endMS = None
						for line in tsvcnt:
							if line[-1] == '\n':
								line = line[:-1]
							tsvTabs = line.split('\t')
							
							if isStartBeat == False:
								if int(tsvTabs[2]) == 1:
									isStartBeat = True
									beatIndex = 1
									startMS = int(float(tsvTabs[0]) * 1000)
							else:
								if int(tsvTabs[2]) == beatIndex + 1:
									beatIndex = beatIndex + 1
									if beatIndex == 4:
										endMS = int(float(tsvTabs[1]) * 1000)
										beatIndex = 0
										
										# Splice and store heartbeat as WAV
										heartsound = record[startMS:endMS]
										if pData["av"]["mr"] == True:
											output_filename = os.path.join("./" + proc_dir + "/" + av_prefix + "_" + murmur_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										else:
											output_filename = os.path.join("./" + proc_dir + "/" + av_prefix + "_" + healthy_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										heartsound.export(output_filename, format="wav")
										
										soundIndex = soundIndex + 1
										
									if beatIndex == 1:
										startMS = int(float(tsvTabs[0]) * 1000)
								else:
									beatIndex = 0
									
				if pData["tv"]["wav"] is not None and pData["tv"]["tsv"] is not None:
					with open(os.path.join("./" + data_dir, pData["tv"]["tsv"])) as tsvfile:
						tsvcnt = tsvfile.readlines()
						tsvfile.close()
						
						record = AudioSegment.from_wav(os.path.join("./" + data_dir, pData["tv"]["wav"]))
						
						isStartBeat = False
						beatIndex = 0
						soundIndex = 0
						startMS = None
						endMS = None
						for line in tsvcnt:
							if line[-1] == '\n':
								line = line[:-1]
							tsvTabs = line.split('\t')
							
							if isStartBeat == False:
								if int(tsvTabs[2]) == 1:
									isStartBeat = True
									beatIndex = 1
									startMS = int(float(tsvTabs[0]) * 1000)
							else:
								if int(tsvTabs[2]) == beatIndex + 1:
									beatIndex = beatIndex + 1
									if beatIndex == 4:
										endMS = int(float(tsvTabs[1]) * 1000)
										beatIndex = 0
										
										# Splice and store heartbeat as WAV
										heartsound = record[startMS:endMS]
										if pData["tv"]["mr"] == True:
											output_filename = os.path.join("./" + proc_dir + "/" + tv_prefix + "_" + murmur_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										else:
											output_filename = os.path.join("./" + proc_dir + "/" + tv_prefix + "_" + healthy_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										heartsound.export(output_filename, format="wav")
										
										soundIndex = soundIndex + 1
										
									if beatIndex == 1:
										startMS = int(float(tsvTabs[0]) * 1000)
								else:
									beatIndex = 0
									
				if pData["mv"]["wav"] is not None and pData["mv"]["tsv"] is not None:
					with open(os.path.join("./" + data_dir, pData["mv"]["tsv"])) as tsvfile:
						tsvcnt = tsvfile.readlines()
						tsvfile.close()
						
						record = AudioSegment.from_wav(os.path.join("./" + data_dir, pData["mv"]["wav"]))
						
						isStartBeat = False
						beatIndex = 0
						soundIndex = 0
						startMS = None
						endMS = None
						for line in tsvcnt:
							if line[-1] == '\n':
								line = line[:-1]
							tsvTabs = line.split('\t')
							
							if isStartBeat == False:
								if int(tsvTabs[2]) == 1:
									isStartBeat = True
									beatIndex = 1
									startMS = int(float(tsvTabs[0]) * 1000)
							else:
								if int(tsvTabs[2]) == beatIndex + 1:
									beatIndex = beatIndex + 1
									if beatIndex == 4:
										endMS = int(float(tsvTabs[1]) * 1000)
										beatIndex = 0
										
										# Splice and store heartbeat as WAV
										heartsound = record[startMS:endMS]
										if pData["mv"]["mr"] == True:
											output_filename = os.path.join("./" + proc_dir + "/" + mv_prefix + "_" + murmur_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										else:
											output_filename = os.path.join("./" + proc_dir + "/" + mv_prefix + "_" + healthy_suffix, str(pData["pID"]) + "_" + str(soundIndex) + ".wav")
										heartsound.export(output_filename, format="wav")
										
										soundIndex = soundIndex + 1
										
									if beatIndex == 1:
										startMS = int(float(tsvTabs[0]) * 1000)
								else:
									beatIndex = 0
								
				print(str(pData))
				
if __name__ == '__main__':
	main()
