# count_lego.py
def count_legos(detections, confidence_threshold=0.5):
    count = 0
    for score in detections['scores']:
        if score > confidence_threshold:
            count += 1
    return count