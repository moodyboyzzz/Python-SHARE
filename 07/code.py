class SpeedViolatorDetector:
    def __init__(self, threshold_speed=40):
        self.threshold_speed = threshold_speed
        self.cars = {}
        self.violators = set()
        
    def process_line(self, line):
        timestamp, track_id, object_type, x, y, city_name = map(str.strip, line.split(','))
        x, y = float(x), float(y)
        timestamp = float(timestamp)
        if track_id not in self.cars:
            self.cars[track_id] = {'prev_timestamp': timestamp, 'prev_x': x, 'prev_y': y, 'sumtime' : 0}
        else:
            speed = ((x - self.cars[track_id]['prev_x'])**2 + (y - self.cars[track_id]['prev_y'])**2)**0.5 / (timestamp - self.cars[track_id]['prev_timestamp']) * 3.6
            self.cars[track_id]['sumtime'] +=  (timestamp - self.cars[track_id]['prev_timestamp'])
            self.cars[track_id]['prev_timestamp'] = timestamp
            self.cars[track_id]['prev_x'] = x
            self.cars[track_id]['prev_y'] = y

            if speed > self.threshold_speed :
                if self.cars[track_id]['sumtime'] >= 1 and not(track_id in self.violators):
                    self.violators.add(track_id)
                    return f"{track_id}"
            else:
                self.cars[track_id]['sumtime'] = 0

    def detect_speed_violators(self, file_path):
        with open('data.csv', 'r') as file:
            next(file)
            for line in file:
                result = self.process_line(line)
                if result:
                    yield result

detector = SpeedViolatorDetector()

with open('answer.txt', 'w') as output_file:
    for violator in detector.detect_speed_violators(file_path):
        output_file.write(violator + '\n')