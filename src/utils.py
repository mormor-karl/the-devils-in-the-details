"""
This file has utility functions that we expect to reuse in different experiments throughout the
whole project.
"""

# Package imports
import logging
import json
import os
from tqdm import tqdm  # my addition

####################################################################################################

def read_swell_directory(path:str="."):
    """
    This function takes a path where you expect to find the directories with the Swell files and
    reads all of the files in there.

    NOTE: If you're expecting to have extra folders or anything like that, you _will_ have to change
    this function to fix some hardcoded assumptions.


    INPUT:

        - path  A string containing the path to the folder containing the folders with the Swell
                files. So, for example if the current folder has "spIn_v2.0", "SW1203_v2.0", and
                "TISUS_v2.0" you would pass the value "." as an argument.
                Default: "."


    OUTPUT:

        - document_list     A list containing a dictionary for each document. The keys for each
                            document are the following:
                            - id                The Swell ID of the imported file
                            - text              The original text of the essay
                            - normalized_text   The normalized essay text
                            - metadata          The metadata imported as a dictionary
                            - svala_graph       The graph for Svala annotations for the current file

        - file_errors       A list containing the IDs of the files where we encountered errors
    """

    # Initialize list of documents and of errors
    document_list = []
    file_errors = []

    # Go through all directories in the current path
    for dir in os.walk(path):

        # Get path parts
        parent = dir[0]
        child = dir[1]
        files = dir[2]

        # Verify we're where we want to be at
        # NOTE: these paths were chosen assuming that you gave the argument "." as a path
        #       if the program breaks when giving it another path, you have to fix this
        if parent in [ path , path+"/__pycache__" ]:      #[".","./__pycache__"]:
            continue

        # NOTE: this was assuming that you do not change the folders for Swell. Things will break
        #       if either you have extra folders within the same Swell folders or if there are
        #       any empty folders
        if len(child) != 0:
            raise Exception("Found a folder in one of the Swell directories")
        if len(files) == 0:
            raise Exception("Found an empty folder in the given path")
        
        # Read current file
        for file in tqdm(files, desc='Reading...'):  # my change
            if not file.startswith('._'):  # my addition here
                curr_path = parent + "/" + file
                document, error = read_swell_file(curr_path)
                if error:
                    file_errors.append(document["id"])
                    continue
                document_list.append(document)
        
    return document_list , file_errors

####################################################################################################

# This reads a single Swell-Pilot file
def read_swell_file(path:str):

    """
    This function reads a single file given its path.

    
    INPUT:

        path    A string containing the path to the file we want to read. The file has to be in the
                format of the files that Elena and Maria worked on (i.e. Swell-Pilot).


    OUTPUT:

        document        A dictionary with the following keys:
                        - id                The Swell ID of the imported file
                        - text              The original text of the essay
                        - normalized_text   The normalized essay text
                        - metadata          The metadata imported as a dictionary
                        - svala_graph       The graph for Svala annotations for the current file

        error_flag      A boolean flag to note whether there were any errors when reading the file
    """

    # Activate logger
    logger = logging.getLogger("swell_read_file")

    # Initialize stuff
    document = {}
    graph_flag = 0
    error_flag = 0

    # Open the file
    with open(path,"r", encoding="utf-8") as F:

        # The documents have very weird formatting
        lines = F.readlines()

        for line in lines:

            # Remove leading/preceding newlines and whitespaces
            line = line.strip("\n ")

            # Skip empty lines
            if len(line) == 0:
                continue

            # Once the Svala Graph starts, ignore everything else
            elif line[:11].lower() == "svala-graph":
                document["svala_graph"] = line.split(":")[1].strip(" ")
                graph_flag = 1

            # This should go before the previous line for computational purposes
            # But I added it here to make things clearer
            elif graph_flag == 1:
                document["svala_graph"] += line

            # ID is easy to get
            elif line[:8].lower() == "essay id":
                document["id"] = line.split(":")[1].strip(" ")

            # Metadata is annowying and I'm sure this will have lots of issues
            elif line[:8].lower() == "metadata":
                items = line[9:].split()
                items_dict = {}
                old_key = ""
                for item in items[1:]:
                    if "=" in item:
                        key, value = item.split('="')
                        items_dict[key] = value.strip('"')
                        old_key = key
                    else:
                        items_dict[old_key] += " " + item
                document["metadata"] = items_dict

            # Getting the original text
            elif line[:6].lower() == "source":
                document["text"] = line[7:].strip()

            # Getting the normalized text
            elif line[:6].lower() == "target":
                document["text_normalized"] = line[7:].strip()

            # If there's a line we don't know how to read, log a warning and raise the error flag
            else:
                error_flag = 1
                logger.warning("Weird line found in document "+document["id"])
                logger.warning("Content: "+line)
                logger.warning(len(line))

        # Read the Svala graph
        try:
            document["svala_graph"] = json.loads(document["svala_graph"])

        # If we cannot read the Svala graph, log the error and raise the error flag
        except json.JSONDecodeError as E:
            error_flag = 1
            logger.error("Could not import svala graph from document "+document["id"])
            logger.error(E)

    return document, error_flag

####################################################################################################
