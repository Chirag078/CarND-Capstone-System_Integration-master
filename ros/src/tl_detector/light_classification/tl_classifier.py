from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf

import rospy



class TLClassifier(object):
    def __init__(self, model_path=None):

        self.graph = self.load_inference_graph(model_path)
        self.sess = tf.Session(graph=self.graph)


    def load_inference_graph(self, path):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        return graph
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        np_image = self.load_image_into_numpy_array(image)
        image_expanded = np.expand_dims(np_image, axis=0)
        with self.graph.as_default():
            #with tf.Session(graph=self.graph) as sess:
            image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
            detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num) = self.sess.run(
                [detect_boxes, detect_scores, detect_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
                
                
        return self.transform_idx(classes[0][0])
    	
    
    def transform_idx(self,idx):
        if idx==1:
            return TrafficLight.GREEN
        if idx==2: 
            return TrafficLight.RED
        if idx ==3:
            return TrafficLight.YELLOW
        if idx ==4:
	    return TrafficLight.UNKNOWN
	return TrafficLight.UNKNOWN

    def load_image_into_numpy_array(self,image):
 
        np_arr = np.asarray(image[:,:])
        return np_arr