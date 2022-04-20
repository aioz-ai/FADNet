"""
Generate .html file with world map and positions of workers and links used in the overlay
"""
import argparse
import os
import time
import mplleaflet
import matplotlib.pyplot as plt
import networkx as nx
from geopy.geocoders import Nominatim


geolocator = Nominatim(user_agent="delay", timeout=20)

parser = argparse.ArgumentParser()
parser.add_argument(
    'underlay',
    help='name of the underlay network; should be present in "/data"',
    type=str)
parser.add_argument(
    'architecture',
    help='name of the architecture; should be present in "results/$UNDERLAY"',
    type=str)

parser.set_defaults(user=False)

args = parser.parse_args()

if __name__ == "__main__":
    underlay_path = os.path.join("data", "{}.gml".format(args.underlay))
    overlay_path = os.path.join("results", args.underlay, "{}.gml".format(args.architecture))

    underlay = nx.read_gml(underlay_path)

    pos_dict = {}
    for node in underlay.nodes():
        try:
            pos_dict[node] = [underlay.nodes(data=True)[node]["Longitude"],
                              underlay.nodes(data=True)[node]["Latitude"]]

        except KeyError:
            time.sleep(1.2)  # To avoid Service time out Error

            geo = geolocator.geocode(node, timeout=20)
            pos_dict[node] = [geo.longitude, geo.latitude]

    overlay = nx.read_gml(overlay_path).to_undirected()

    mapping = {}
    for ii, node in enumerate(underlay.nodes()):
        mapping[str(ii)] = node

    overlay = nx.relabel_nodes(overlay, mapping).to_undirected()

    fig, ax = plt.subplots()

    nx.draw_networkx_nodes(overlay, pos=pos_dict, node_size=10, node_color='red', edge_color='k', alpha=.5,
                           with_labels=True)
    nx.draw_networkx_edges(overlay, pos=pos_dict, edge_color='blue', alpha=1, width=5.0)
    nx.draw_networkx_labels(overlay, pos=pos_dict, label_pos=10.3)

    mplleaflet.display(fig=ax.figure)
    mplleaflet.save_html(fig=ax.figure,
                         fileobj=os.path.join("results", args.underlay, "{}.html".format(args.architecture)))
