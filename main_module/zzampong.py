# USAGE
# python zzampong.py --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer_default.pickle --le output/le_default.pickle --recognizer_mask output/recognizer_mask.pickle --le_mask output/le_mask.pickle

# import the necessary packages
from functions import dectecting_functions as df
from functions import recognizing_functions as rf
from functions import fusion_functions as ff

from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import pickle
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--accessory_model", type=str,
	default="accessory_detector.model",
	help="path to trained accessory detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

ap.add_argument("-e", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")

ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, default="output/le.pickle",
	help="path to label encoder")

ap.add_argument("-re", "--recognizer_mask", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-le", "--le_mask", required=True,
	help="path to label encoder")

args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face accesory detector model...")
accessoryNet = load_model(args["accessory_model"])

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

recognizer_mask = pickle.loads(open(args["recognizer_mask"], "rb").read())
le_mask = pickle.loads(open(args["le_mask"], "rb").read())

print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

default_data = {"name":['JaeYoon', 'Minsik', 'YoungKwang', 'irin'], "count":[0.8, 0.8, 0.8, 0.8], "best":[0.8, 0.8, 0.8, 0.9], "avg":[0.6, 0.6, 0.6, 0.8]}

fullface = {"name":[], "count":[], "best":[], "avg":[], "score":[]}
maskedface = {"name":[], "count":[], "best":[], "avg":[], "score":[]}
sunglassedface = {"name":[], "count":[], "best":[], "avg":[], "score":[]}
acc_count = {"acc_list": (fullface,maskedface,sunglassedface), "acc_name":("no accessory","mask", "sunglass"), "count":[0,0,0]}
recorded_time = {"name":[], "time":[]}

for person in le.classes_:
	fullface["name"].append(person)
	fullface["count"].append(0)
	fullface["avg"].append(0)
	fullface["best"].append(0)
	fullface["score"].append(0)
for person in le_mask.classes_:
	maskedface["name"].append(person)
	maskedface["count"].append(0)
	maskedface["avg"].append(0)
	maskedface["best"].append(0)
	maskedface["score"].append(0)
face_count = 0

#-------------------------------------------------------------------------------------------------------------------------- input data

while True:
	frame = vs.read()
	#frame = imutils.resize(frame, width=600)

	preds_accessory = df.detect_and_predict_accesory(frame, faceNet, accessoryNet)

	for pred in preds_accessory:
		(default, mask, sunglass) = pred
		start_time = time.time()

		if max(default, mask, sunglass)==mask:
			rf.face_recognition(frame, faceNet, embedder, recognizer_mask,le_mask, maskedface,recorded_time,"mask")
		elif max(default, mask, sunglass)==default:
			rf.face_recognition(frame, faceNet, embedder, recognizer,le, fullface,recorded_time,"default")

		face_count += 1
	if face_count > 99:
		break
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()

acc_count["count"][0] = sum(fullface["count"])
acc_count["count"][1] = sum(maskedface["count"])
acc_count["count"][2] = sum(sunglassedface["count"])

pred_acc = acc_count["acc_list"][acc_count["count"].index(max(acc_count["count"]))]
result = acc_count["acc_name"][acc_count["count"].index(max(acc_count["count"]))]
print("\n[RESULT] The predicted accessory is", result)

#-------------------------------------------------------------------------------------------------------------------------- default first score

if result == "no accessory":
	for num in pred_acc["count"]:
		if num != 0:
			pred_acc["avg"][pred_acc["count"].index(num)] = pred_acc["avg"][pred_acc["count"].index(num)] / num
		pred_acc["count"][pred_acc["count"].index(num)] = num / sum(acc_count["count"])

	pred_acc["score"][pred_acc["count"].index(max(pred_acc["count"]))] += 1
	pred_acc["score"][pred_acc["best"].index(max(pred_acc["best"]))] += 1
	pred_acc["score"][pred_acc["avg"].index(max(pred_acc["avg"]))] += 1

	for i in range(0,len(pred_acc["name"])-1):
		if pred_acc["count"][i] > default_data["count"][i]:
			pred_acc["score"][i] += 2
		if pred_acc["avg"][i] > default_data["avg"][i]:
			pred_acc["score"][i] += 2
		if pred_acc["best"][i] > default_data["best"][i]:
			pred_acc["score"][i] += 2

#-------------------------------------------------------------------------------------------------------------------------- mask first score

elif result == "mask":
	for avg in pred_acc["avg"]:
		pred_acc["avg"][pred_acc["avg"].index(avg)] = pred_acc["avg"][pred_acc["avg"].index(avg)]/sum(acc_count["count"])
	for avg in pred_acc["avg"]:
		if pred_acc["count"][pred_acc["avg"].index(avg)] != 0:
			if avg > 0.2:
				pred_acc["score"][pred_acc["avg"].index(avg)] += 1
	for num in pred_acc["count"]:
		pred_acc["count"][pred_acc["count"].index(num)] = num / sum(acc_count["count"])
	for num in pred_acc["count"]:
		if num > 0.2:
			pred_acc["score"][pred_acc["count"].index(num)] += 1

	pred_acc["score"][pred_acc["count"].index(max(pred_acc["count"]))] += 1
	pred_acc["score"][pred_acc["best"].index(max(pred_acc["best"]))] += 1
	pred_acc["score"][pred_acc["avg"].index(max(pred_acc["avg"]))] += 1

#-------------------------------------------------------------------------------------------------------------------------- get first, second user

print(pred_acc)
first_user = pred_acc["name"].pop(pred_acc["score"].index(max(pred_acc["score"])))
first_user_avg = pred_acc["avg"].pop(pred_acc["score"].index(max(pred_acc["score"])))
first_user_score = pred_acc["score"].pop(pred_acc["score"].index(max(pred_acc["score"])))
first_user = first_user.split("_")[0]

second_user_score = 0
secondscore_idx = 0
for score in pred_acc["score"]:
	if score >= second_user_score:
		second_user = pred_acc["name"][secondscore_idx]
		second_user_avg = pred_acc["avg"][secondscore_idx]
		second_user_score = score
	secondscore_idx += 1
second_user = second_user.split("_")[0]

print("first:{}(score:{})".format(first_user, first_user_score))
print("second:{}(score:{})".format(second_user, second_user_score))
first_final = {"name": [first_user, second_user], "avg": [first_user_avg,second_user_avg]}
second_final = {"name": [first_user, second_user], "avg": [0,0], "best":[0,0], "count": [0,0], "wrong":[0,0], "total":[0,0], "score": [1,0]}

#-------------------------------------------------------------------------------------------------------------------------- default configure lock/unlock

if result == "no accessory":
	if first_user_score > 5:
		status = "unlock"
	else:
		status = "lock"
	print("[Result] User is {}(status : {})".format(first_user, status))

#--------------------------------------------------------------------------------------------------------------------------  mask 합성 모델 생성

elif result == "mask":
	for names in second_final["name"]:
		N = 0; 	cnt = [i for i in range(1, 5)]
		while N < 20:
			if names != "unknown" :
				N, cnt = ff.mask_fuse(second_final,names,cnt,N)
			elif names == "unknown" :
				N = 21
			if len(cnt) ==0:
				N=21

	rf.second_recognition(faceNet, embedder, recognizer, le, second_final)

	for count in second_final["count"]:
		if count != 0:
			second_final["avg"][second_final["count"].index(count)] = second_final["avg"][second_final["count"].index(count)]/count
	second_final["score"][second_final["best"].index(max(second_final["best"]))] += 1
	second_final["score"][second_final["avg"].index(max(second_final["avg"]))] += 1

	if second_final["avg"][0] - first_final["avg"][0] > second_final["avg"][1] - first_final["avg"][1]:
		if max(second_final["avg"]) == second_final["avg"][0]:
			second_final["score"][0] += 1
	else:
		if max(second_final["avg"]) == second_final["avg"][1]:
			second_final["score"][1] += 1

	for i in range(0,2):
		if second_final["total"][i] != 0 :
			second_final["wrong"][i] = second_final["wrong"][i]/second_final["total"][i]
	if 0 not in second_final["total"]:
		if 0 != second_final["wrong"][0] and 0 != second_final["wrong"][1]:
			thres = 4
			if second_final["wrong"][0] > second_final["wrong"][1]:
				second_final["score"][1] += 1

				if second_final["wrong"][1] == 0:
					second_final["score"][1] += 1
			else:
				second_final["score"][0] += 1
				if second_final["wrong"][0] == 0:
					second_final["score"][0] += 1
		else:
			thres = 4
	else:
		thres = 3
	print("[first_final]\n", first_final)
	print("[second_final]\n", second_final)

	# -------------------------------------------------------------------------------------------------------------------------- mask configure lock/unlock

	if "unknown" in second_final["name"]:
		if max(second_final["score"]) > thres :
			status = "unlock"
			cv2.putText(frame, "unlock", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 7)
		else:
			status = "lock"
			cv2.putText(frame, "lock", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 7)
	else:
		if max(second_final["score"]) > thres :
			status = "unlock"
			cv2.putText(frame, "unlock", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 7)
		else:
			status = "lock"
			cv2.putText(frame, "lock", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 7)
	print("[Result] User is {}(status : {})".format(second_final["name"]
													[second_final["score"].index(max(second_final["score"]))], status))
