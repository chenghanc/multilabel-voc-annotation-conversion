import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

input_img_folder =  'head-conversion/images'
input_ann_folder =  'head-conversion/annotations'
output_ann_folder = 'head-conversion/annotations_xml'
output_img_folder = 'head-conversion/images_xml'

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_ann_folder, exist_ok=True)

image_list = os.listdir(input_img_folder)
annotation_list = os.listdir(input_ann_folder)

label_dict = {
	#"Head"         : "Head",
	"blue"         : "blue",
	"yellow"       : "yellow",
	"white"        : "white",
	"red"          : "red",
	"AdultHead"    : "AdultHead",
	"RealBabyHead" : "RealBabyHead",
	"mask"         : "mask",
	"unmask"       : "unmask"
}

classes = ["blue", "yellow", "white", "red", "AdultHead", "RealBabyHead", "mask", "unmask"]
thickness = 2
color = (255,0,0)
count = 0

def object_string(label, bbox):
	req_str = '''
	<object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>'''.format(label, bbox[0], bbox[1], bbox[2], bbox[3])
	return req_str

def object_string_head(label_head, bbox):
	req_str_head = '''
	<object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>'''.format(label_head, bbox[0], bbox[1], bbox[2], bbox[3])
	return req_str_head

for annotation in annotation_list:
	annotation_path = os.path.join(os.getcwd(), input_ann_folder, annotation)
	xml_annotation = annotation.split('.xml')[0] + '.xml'
	xml_path = os.path.join(os.getcwd(), output_ann_folder, xml_annotation)
	img_file = annotation.split('.xml')[0] + '.jpg'
	img_path = os.path.join(os.getcwd(), input_img_folder, img_file)
	output_img_path = os.path.join(os.getcwd(), output_img_folder, img_file)
	img = cv2.imread(img_path)
	annotation_string_init = '''<annotation>
	<folder>annotations</folder>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>{}</depth>
	</size>
	<segmented>0</segmented>'''.format(img_file, img_path, img.shape[1], img.shape[0], img.shape[2])

	file = open(annotation_path, 'r')
	lines = file.read()
	tree=ET.parse(annotation_path)
	root = tree.getroot()
	#print(root.tag)
	#print(root.attrib)

	#for child in root:
	#	print(child.tag, child.attrib)
	
	for obj in root.iter('object'):
		difficult = obj.find('difficult').text
		cls = obj.find('name').text
		if cls not in classes:
		    continue
		cls_id = classes.index(cls)
		
		xmlbox = obj.find('bndbox')
		b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
		xmin = b[0]
		ymin = b[1]
		xmax = b[2]
		ymax = b[3]
		print(cls_id, xmin, ymin, xmax, ymax, xml_annotation)
		new_coords_min = (int(xmin), int(ymin))
		new_coords_max = (int(xmax), int(ymax))
		bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
		label = label_dict.get(cls)
		label_head = "Head"
		req_str = object_string(label, bbox)
		req_str_head = object_string_head(label_head, bbox)
		#annotation_string_init = annotation_string_init + req_str
		annotation_string_init = annotation_string_init + req_str + req_str_head
		cv2.rectangle(img, new_coords_min, new_coords_max, color, thickness)
	cv2.imwrite(output_img_path, img)
	annotation_string_final = annotation_string_init + '\n' + '</annotation>'
	f = open(xml_path, 'w')
	f.write(annotation_string_final)
	f.close()
	count += 1
	print('[INFO] Completed {} image(s) and annotation(s) pair'.format(count))