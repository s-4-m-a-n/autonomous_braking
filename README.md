# autonomous braking :red_car:

It is a computer vision project that uses yolov5 for object detection. It estimates distance between the subject (vehicle where a dash cam and the system is installed) and the traffic objects, if the estimated distance exceeds the predetermined threshold distance than a braking signal will be generated.  

# Importance of autonomous braking

- Only 40% of the drivers apply brake in an accidental situation.
- Laboratory test results have proven that about 47% of the collision impact can be reduced by automatic application of the brake force.
 
# Screenshots
- when a traffic object is within the ROI but outside the threshold distance --> braking signal is not generated
![screenshot 1](https://github.com/s-4-m-a-n/autonomous_braking/blob/master/screenshots/Screenshot_1.png)

- when a traffic object is within the ROI and below the threshold distance --> braking signal is generated
![screenshot 2](https://github.com/s-4-m-a-n/autonomous_braking/blob/master/screenshots/Screenshot_2.png)

- when a traffic object is outside the ROI but below the threshold distance --> braking signal is not generated
![screenshot 2](https://github.com/s-4-m-a-n/autonomous_braking/blob/master/screenshots/Screenshot_3.png)

