"""Program that reads raw IAM handwriting data and writes out
serialized dictionaries containing the parsed data that are used to
train the neural network."""

from __future__ import division
from __future__ import print_function
import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as etree

FLAGS = None
TARGET_CHARS_FILE = "/Users/zacwilson/data/iam_handwriting/characters.json"
with open(TARGET_CHARS_FILE, "r") as filein:
    TARGET_CHARS = json.load(filein)

def parse_input_file(filename):
    """Parses an input XML file and returns the parsed sequence example.

    Args:
        filename: Name of the input XML file to parse.

    Returns:
        A dictionary containing the parsed sequence of handwriting
        data, as well as the corresponding label sequence. The
        dictionary contains the followsing fields:
        labels: 
        seq_length: The length of the input data sequence
        x_coordinates: the x coordinates of the pen
        y_coordinates: the y coordinates of the pen
        (Note: the x and y coordinates are relative to the upper left
        corner of the smartboard, which is considered the origin)
        timestamps: timestamps for each datapoint in the input sequence
        is_stroke_start: vector of boolean flags indicating whether
            each datapoint in the sequence is the start of a pen stroke
        is_stroke_end: vector of boolean flags indicating whether
            each datapoint in the sequence is the start of a pen stroke
        as boolean flags indicating whether the datapoint corresponds
        to the beginning or the end of a stroke. More concretely, here
        are the set of dictionaries that one would use to parse a
        serialized sequence example:

    Raises:
        IOError: If an IO error occurs when attempting to read the
            input XML file or write the output tfrecord file.
        ValueError: If the contents of the input XML file do not conform
            to the expectations about what the format of the input data
            should be.
    """
    # parse the xml file
    filepath = os.path.join(FLAGS.input_dir, filename)
    tree = etree.parse(filepath)
    root = tree.getroot()

    # verify that corner attribute of SensorLocation element is "top_left"
    sensor_location_list = root.findall("*/SensorLocation")
    if len(sensor_location_list) != 1:
        raise ValueError("Expected number of SensorLocation elements in XML " +
                         "file is one.\nActual number of SensorLocation " +
                         "elements was {0}\nOffending XML file: {1}"
                         .format(len(sensor_location_list), filepath))
    sensor_location = sensor_location_list[0]
    if not sensor_location.attrib.has_key("corner"):
        raise ValueError("SensorLocation element does not have \"corner\" " +
                         "attribute.\nOffending XML file: {0}"
                         .format(filepath))
    corner = sensor_location.attrib["corner"]
    if corner != "top_left":
        raise ValueError("\"corner\" attribute of SensorLocation element is " +
                         "expected to have value \"top_left\".\nActual " +
                         "value: {0}\nOffending XML file: {1}"
                         .format(corner, filepath))

    # get x coordinate of the top left corner (of the line)
    vert_opp_coords_list = root.findall("*/VerticallyOppositeCoords")
    if len(vert_opp_coords_list) != 1:
        raise ValueError("Expected number of VerticallyOppositeCoords " +
                         "elements in XML file is one.\nActual number of " +
                         "VerticallyOppositeCoords elements was {0}\n" +
                         "Offending XML file: {1}"
                         .format(len(vert_opp_coords_list), filepath))
    vert_opp_coords = vert_opp_coords_list[0]
    if not vert_opp_coords.attrib.has_key("x"):
        raise ValueError("VerticallyOppositeCoords element does not have " +
                         "\"x\" attribute.\nOffending XML file: {0}"
                         .format(filepath))
    top_left_x = int(vert_opp_coords.attrib["x"])

    # get x coordinate of the top left corner (of the line)
    horiz_opp_coords_list = root.findall("*/HorizontallyOppositeCoords")
    if len(horiz_opp_coords_list) != 1:
        raise ValueError("Expected number of HorizontallyOppositeCoords " +
                         "elements in XML file is one.\nActual number of " +
                         "HorizontallyOppositeCoords elements was {0}\n" +
                         "Offending XML file: {1}"
                         .format(len(horiz_opp_coords_list), filepath))
    horiz_opp_coords = horiz_opp_coords_list[0]
    if not horiz_opp_coords.attrib.has_key("y"):
        raise ValueError("HorizontallyOppositeCoords element does not have " +
                         "\"y\" attribute.\nOffending XML file: {0}"
                         .format(filepath))
    top_left_y = int(horiz_opp_coords.attrib["y"])

    # get list of all strokes
    stroke_list = root.findall("*/Stroke")
    # verify that there is at least one stroke in XML file
    if len(stroke_list) == 0:
        raise ValueError("XML file {0} does not contain any strokes"
                         .format(filepath))

    # get the start time of the first stroke in the line
    first_stroke = stroke_list[0]
    if not first_stroke.attrib.has_key("start_time"):
        raise ValueError("First stroke does not contain a \"start_time\" " +
                         "attribute.\nOffending XML file: {0}"
                         .format(filepath))
    line_start_time = float(first_stroke.attrib["start_time"])

    # lists to store parsed input sequence data:
    x_coordinates = []
    y_coordinates = []
    timestamps = []
    is_stroke_start = []
    is_stroke_end = []

    # parse data from all strokes in XML file
    for stroke in stroke_list:
        # verify that stroke begins after line_start_time
        if not stroke.attrib.has_key("start_time"):
            error_msg = ("Stroke element does not contain a " +
                         "\"start_time\" attribute.\nOffending " +
                         "XML file: {0}")
            raise ValueError(error_msg.format(filepath))
        if float(stroke.attrib["start_time"]) < line_start_time:
            index = stroke_list.index(stroke)
            stroke_start_time = stroke.attrib["start_time"]
            error_msg = ("Start time of first stroke in XML file ({0}) " +
                         "is apparently later than start time of stroke " +
                         "number {1}, which is {2}.\nOffending XML file: " +
                         "{3}")
            raise ValueError(error_msg.format(line_start_time, index,
                                              stroke_start_time, filepath))
        points = stroke.findall("Point")
        if len(points) == 0:
            error_msg = ("Encountered stroke with no points.\nOffending " +
                         "XML file: {0}")
            raise ValueError(error_msg.format(filepath))
        # append parsed data from first point in stroke
        x, y, timestamp = parse_point(points[0], filepath)
        x_coordinates.append(x - top_left_x)
        y_coordinates.append(y - top_left_y)
        timestamps.append(timestamp - line_start_time)
        is_stroke_start.append(1)
        is_stroke_end.append(0)
        # append parsed data from middle points in stroke
        for point in points[1:-1]:
            x, y, timestamp = parse_point(point, filepath)
            x_coordinates.append(x - top_left_x)
            y_coordinates.append(y - top_left_y)
            timestamps.append(timestamp - line_start_time)
            is_stroke_start.append(0)
            is_stroke_end.append(0)
        # append parsed data from last point in stroke
        x, y, timestamp = parse_point(points[-1], filepath)
        x_coordinates.append(x - top_left_x)
        y_coordinates.append(y - top_left_y)
        timestamps.append(timestamp - line_start_time)
        is_stroke_start.append(0)
        is_stroke_end.append(1)

    # return parsed data in dictionary
    results = {}
    results["seq_length"] = len(x_coordinates)
    results["x_coordinates"] = x_coordinates
    results["y_coordinates"] = y_coordinates
    results["timestamps"] = timestamps
    results["is_stroke_start"] = is_stroke_start
    results["is_stroke_end"] = is_stroke_end
    return results


def parse_point(point, filepath):
    """Parses the x, y, and timestamp attributes from an XML Point element."""
    # verify that point contains all required attributes
    if not point.attrib.has_key("x"):
        raise ValueError("Point element does not have \"x\" attribute.\n" +
                         "Offending XML file: {0}".format(filepath))
    if not point.attrib.has_key("y"):
        raise ValueError("Point element does not have \"y\" attribute.\n" +
                         "Offending XML file: {0}".format(filepath))
    if not point.attrib.has_key("time"):
        raise ValueError("Point element does not have \"time\" attribute.\n" +
                         "Offending XML file: {0}".format(filepath))
    x = int(point.attrib["x"])
    y = int(point.attrib["y"])
    timestamp = float(point.attrib["time"])
    return x, y, timestamp


def parse_label_file(filename):
    """Parses a file containing target label sequences.

    Args:
        filename: Name of label file to parse.

    Returns:
        A list of label sequences for each line in the label file.
        each line's label sequence is a sequence of integers
        corresponding to the index of the characters in the line in the
        global TARGET_CHARS list.

    Raises:
        IOError: If an IOError occurs trying to read the input file.
        ValueError: If the input label file does not conform to
            expectations about how a label file should be formatted, or
            what values it can legally contain.
    """
    line_labels = []
    with open(filename, "r") as filein:
        line = filein.readline()
        while True:
            if not line:
                error_msg = "Label file {0} does not contain a CSR section"
                raise ValueError(error_msg.format(filename))
            if line.strip() == "CSR:":
                break
            line = filein.readline()
        line = filein.readline().strip()
        if line != "":
            error_msg = ("Expected a blank line after CSR section header.\n" +
                         "Offending label file: {0}")
            raise ValueError(error_msg.format(label_file))
        line = filein.readline().strip()
        while line:
            label = []
            line = line.strip()
            for char in line:
                if char not in TARGET_CHARS:
                    error_msg = ("The character {0}, which was found in " +
                                 "label file {1}, is not present in set of " +
                                 "TARGET_CHARS")
                    raise ValueError(error_msg.format(char, filename))
                label.append(TARGET_CHARS.index(char))
            line_labels.append(label)
            line = filein.readline()
    return line_labels


def main():
    """Parses and serializes handwriting recognition examples."""
    training_output_dir = os.path.join(FLAGS.output_dir, "training")
    val_1_output_dir = os.path.join(FLAGS.output_dir, "validation_1")
    val_2_output_dir = os.path.join(FLAGS.output_dir, "validation_2")
    testing_output_dir = os.path.join(FLAGS.output_dir, "testing")

    # PARSE AND SERIALIZE TRAINING DATA
    # logging variables
    num_failed_training_examples = 0
    failed_training_examples = []
    # get list of training examples
    with open(FLAGS.trainset_file, "r") as filein:
        lines = filein.readlines()
    lines = [line.strip() for line in lines]
    # parse training examples and serialize data
    for base_example in lines:
        form_name = base_example.split("-")[0]
        temp = base_example
        if not base_example[-1].isdigit():
            temp = base_example[:-1]
        regex_part1 = os.path.join(FLAGS.input_dir, form_name, temp,
                                   base_example)
        regex = regex_part1 + "-*.xml"
        fields = regex_part1.split(os.path.sep)
        labels_filename = os.path.join(FLAGS.labels_dir, fields[-3],
                                       fields[-2], fields[-1]) + ".txt"
        line_labels = parse_label_file(labels_filename)
        examples = glob.glob(regex)
        for example in examples:
            try:
                parsed_data = parse_input_file(example)
                line_no = int(example.split("-")[-1].rstrip(".xml"))
                if line_no > len(line_labels):
                    error_msg = ("line_no (={0}) of example {1} exceeds " +
                                 "number of line labels parsed in " +
                                 "corresponding labels file {2}")
                    raise ValueError(error_msg.format(line_no, example,
                                     labels_filename))
                parsed_data["labels"] = line_labels[line_no - 1]
                example_name = os.path.basename(example)
                example_name = example_name.rstrip(".xml") + ".json"
                fileout_name = os.path.join(training_output_dir, example_name)
                with open(fileout_name, "w") as fileout:
                    json.dump(parsed_data, fileout)
            except ValueError as error:
                print(error)
                num_failed_training_examples += 1
                failed_training_examples.append(example)            
    print("num_failed_training_examples: {0}"
          .format(num_failed_training_examples))
    print("failed_training_examples: {0}"
          .format(failed_training_examples))

    # PARSE AND SERIALIZE VALIDATION SET ONE DATA
    # logging variables
    num_failed_val_1_examples = 0
    failed_val_1_examples = []
    # get list of validation one examples
    with open(FLAGS.validationset_one_file, "r") as filein:
        lines = filein.readlines()
    lines = [line.strip() for line in lines]
    # parse validation one examples and serialize data
    for base_example in lines:
        form_name = base_example.split("-")[0]
        temp = base_example
        if not base_example[-1].isdigit():
            temp = base_example[:-1]
        regex_part1 = os.path.join(FLAGS.input_dir, form_name, temp,
                                   base_example)
        regex = regex_part1 + "-*.xml"
        fields = regex_part1.split(os.path.sep)
        labels_filename = os.path.join(FLAGS.labels_dir, fields[-3],
                                       fields[-2], fields[-1]) + ".txt"
        line_labels = parse_label_file(labels_filename)
        examples = glob.glob(regex)
        for example in examples:
            try:
                parsed_data = parse_input_file(example)
                line_no = int(example.split("-")[-1].rstrip(".xml"))
                parsed_data["labels"] = line_labels[line_no - 1]
                example_name = os.path.basename(example)
                example_name = example_name.rstrip(".xml") + ".json"
                fileout_name = os.path.join(val_1_output_dir, example_name)
                with open(fileout_name, "w") as fileout:
                    json.dump(parsed_data, fileout)
            except ValueError as error:
                print(error)
                num_failed_val_1_examples += 1
                failed_val_1_examples.append(example)            
    print("num_failed_val_1_examples: {0}"
          .format(num_failed_val_1_examples))
    print("failed_val_1_examples: {0}"
          .format(failed_val_1_examples))

    # PARSE AND SERIALIZE VALIDATION SET TWO DATA
    # logging variables
    num_failed_val_2_examples = 0
    failed_val_2_examples = []
    # get list of validation two examples
    with open(FLAGS.validationset_two_file, "r") as filein:
        lines = filein.readlines()
    lines = [line.strip() for line in lines]
    # parse validation two examples and serialize data
    for base_example in lines:
        form_name = base_example.split("-")[0]
        temp = base_example
        if not base_example[-1].isdigit():
            temp = base_example[:-1]
        regex_part1 = os.path.join(FLAGS.input_dir, form_name, temp,
                                   base_example)
        regex = regex_part1 + "-*.xml"
        fields = regex_part1.split(os.path.sep)
        labels_filename = os.path.join(FLAGS.labels_dir, fields[-3],
                                       fields[-2], fields[-1]) + ".txt"
        line_labels = parse_label_file(labels_filename)
        examples = glob.glob(regex)
        for example in examples:
            try:
                parsed_data = parse_input_file(example)
                line_no = int(example.split("-")[-1].rstrip(".xml"))
                parsed_data["labels"] = line_labels[line_no - 1]
                example_name = os.path.basename(example)
                example_name = example_name.rstrip(".xml") + ".json"
                fileout_name = os.path.join(val_2_output_dir, example_name)
                with open(fileout_name, "w") as fileout:
                    json.dump(parsed_data, fileout)
            except ValueError as error:
                print(error)
                num_failed_val_2_examples += 1
                failed_val_2_examples.append(example)            
    print("num_failed_val_2_examples: {0}"
          .format(num_failed_val_2_examples))
    print("failed_val_2_examples: {0}"
          .format(failed_val_2_examples))

    # PARSE AND SERIALIZE TESTING DATA
    # logging variables
    num_failed_testing_examples = 0
    failed_testing_examples = []
    # get list of training examples
    with open(FLAGS.testset_file, "r") as filein:
        lines = filein.readlines()
    lines = [line.strip() for line in lines]
    # parse training examples and serialize data
    for base_example in lines:
        form_name = base_example.split("-")[0]
        temp = base_example
        if not base_example[-1].isdigit():
            temp = base_example[:-1]
        regex_part1 = os.path.join(FLAGS.input_dir, form_name, temp,
                                   base_example)
        regex = regex_part1 + "-*.xml"
        fields = regex_part1.split(os.path.sep)
        labels_filename = os.path.join(FLAGS.labels_dir, fields[-3],
                                       fields[-2], fields[-1]) + ".txt"
        line_labels = parse_label_file(labels_filename)
        examples = glob.glob(regex)
        for example in examples:
            try:
                parsed_data = parse_input_file(example)
                line_no = int(example.split("-")[-1].rstrip(".xml"))
                parsed_data["labels"] = line_labels[line_no - 1]
                example_name = os.path.basename(example)
                example_name = example_name.rstrip(".xml") + ".json"
                fileout_name = os.path.join(testing_output_dir, example_name)
                with open(fileout_name, "w") as fileout:
                    json.dump(parsed_data, fileout)
            except ValueError as error:
                print(error)
                num_failed_testing_examples += 1
                failed_testing_examples.append(example)            
    print("num_failed_training_examples: {0}"
          .format(num_failed_testing_examples))
    print("failed_training_examples: {0}"
          .format(failed_testing_examples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/zacwilson/data/iam_handwriting/lineStrokes",
        help="Directory from which to read the input sequence data files"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="/Users/zacwilson/data/iam_handwriting/rnnlib-iam-master/ascii",
        help="Directory containing the target label sequences."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/zacwilson/data/iam_handwriting",
        help="Directory in which to write the output tensorflow records"
    )
    parser.add_argument(
        "--trainset_file",
        type=str,
        default="/Users/zacwilson/data/iam_handwriting/rnnlib-iam-master/trainset.txt",
        help="File specifying which examples to use for training."
    )
    parser.add_argument(
        "--testset_file",
        type=str,
        default="/Users/zacwilson/data/iam_handwriting/rnnlib-iam-master/testset_f.txt",
        help="File specifying which examples to use for final testing."
    )
    parser.add_argument(
        "--validationset_one_file",
        type=str,
        default="/Users/zacwilson/data/iam_handwriting/rnnlib-iam-master/testset_v.txt",
        help="File specifying which examples to use for round one validation testing."
    )
    parser.add_argument(
        "--validationset_two_file",
        type=str,
        default="/Users/zacwilson/data/iam_handwriting/rnnlib-iam-master/testset_t.txt",
        help="File specifying which examples to use for round two validation testing."
    )
    FLAGS, _ = parser.parse_known_args()
    main()
