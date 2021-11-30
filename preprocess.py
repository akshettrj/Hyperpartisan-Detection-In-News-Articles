import xml.etree.ElementTree as ET

xmltree_file = './data/articles-training-byarticle-20181122.xml'
tree = ET.parse(xmltree_file)
root = tree.getroot()

for child in root:
    attributes = child.attrib
    