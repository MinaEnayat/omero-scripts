# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # -----------------------------------------------------------------------------
# #   Copyright (C) 2006-2021 University of Dundee. All rights reserved.
# #
# #
# #   This program is free software; you can redistribute it and/or modify
# #   it under the terms of the GNU General Public License as published by
# #   the Free Software Foundation; either version 2 of the License, or
# #   (at your option) any later version.
# #   This program is distributed in the hope that it will be useful,
# #   but WITHOUT ANY WARRANTY; without even the implied warranty of
# #   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# #   GNU General Public License for more details.
# #
# #   You should have received a copy of the GNU General Public License along
# #   with this program; if not, write to the Free Software Foundation, Inc.,
# #   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# #
# # ------------------------------------------------------------------------------

"""
This script converts a Dataset of Images into a Plate, dynamically assigning wells based on image coordinates (X, Y).
"""

import numpy as np
import omero
import omero.scripts as scripts
from omero.gateway import BlitzGateway
from omero.model import WellI, WellSampleI, PlateI, ImageI, PlateAcquisitionI, ScreenI, ScreenPlateLinkI
from omero.rtypes import rint, rlong, rstring, robject, unwrap


def get_image_positions(conn, dataset_id):
    """
    Retrieves the X, Y coordinates of images in a dataset.
    Returns arrays of X, Y coordinates and image IDs.
    """
    xs, ys, image_ids = [], [], []

    dataset = conn.getObject("Dataset", dataset_id)
    if dataset is None:
        return None, None, None

    for image in dataset.listChildren():
        pix_o = image.getPrimaryPixels()
        plane_info = list(pix_o.copyPlaneInfo())[0]  # Get the first plane info
        xs.append(round(plane_info.positionX._value))
        ys.append(round(plane_info.positionY._value))
        image_ids.append(image.getId())
        xs, xy = np.array(xs, dtype="int64"), np.array(ys, dtype="int64")

    return xs, xy, image_ids


def compute_grid_positions(xs, ys, first_well):
    """
    Computes well positions based on image coordinates.
    Uses a 9000-pixel spacing grid and assigns local positions within wells.
    """
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    # Define the bottom-right reference point
    bottom_right_point = (np.max(xs), np.min(ys))

    # Compute distances and filter nearby points
    radius = 3470
    points = np.column_stack((xs, ys))
    distances = np.linalg.norm(points - bottom_right_point, axis=1)
    nearby_indices = np.where(distances <= radius)[0]
    nearby_points = points[nearby_indices]

    # Compute the center of the cluster
    cx = np.mean(nearby_points[:, 0])
    cy = np.mean(nearby_points[:, 1])

    # Compute grid indices for wells
    grid_x = np.round((points[:, 0] - cx) / 9000)
    grid_y = np.round((points[:, 1] - cy) / 9000)
    grid = np.column_stack((grid_x, grid_y))

    # Compute local (internal) coordinates within each well
    internal_x = points[:, 0] - (grid[:, 0] * 9000 + cx)
    internal_y = points[:, 1] - (grid[:, 1] * 9000 + cy)
    grid_local = np.column_stack((internal_x, internal_y))

    # Adjust row/column based on the first well assigned
    offset_row = ord(first_well[0]) - ord('A')
    offset_col = int(first_well[1:]) - 1
    grid[:, 0] *= -1
    grid[:, 0] = (grid[:, 0]) - np.min((grid[:, 0])) + offset_col
    grid[:, 1] = grid[:, 1] - np.min((grid[:, 1])) + offset_row

    return grid, grid_local


def dataset_to_plate(conn, script_params, dataset_id, screen):
    """
    Converts a dataset into a plate, placing images in wells based on coordinates.
    """
    xs, ys, image_ids = get_image_positions(conn, dataset_id)
    if xs is None:
        return None, None, "No valid images found in dataset."

    grid, grid_local = compute_grid_positions(xs, ys, script_params["First_Well"])
    if grid is None:
        return None, None, "Grid computation failed."

    update_service = conn.getUpdateService()
    dataset = conn.getObject("Dataset", dataset_id)

    # Create Plate
    plate = PlateI()
    plate.name = rstring(dataset.name)
    plate.columnNamingConvention = rstring(script_params["Column_Names"])
    plate.rowNamingConvention = rstring(script_params["Row_Names"])
    plate = update_service.saveAndReturnObject(plate)

    if screen:
        link = ScreenPlateLinkI()
        link.parent = ScreenI(screen.id, False)
        link.child = PlateI(plate.id.val, False)
        update_service.saveObject(link)
    else:
        link = None

    plate_acq = PlateAcquisitionI()
    plate_acq.plate = plate
    plate_acq.name = rstring(dataset.name)
    plate_acq = update_service.saveAndReturnObject(plate_acq)

    wells = {}
    for (gx, gy), (lx, ly), img_id in zip(grid, grid_local, image_ids):
        well_label = f"{gx},{gy}"

        if well_label not in wells:
            # Create a new well in OMERO
            well = WellI()
            well.plate = plate
            well.column = rint(int(gx))
            well.row = rint(int(gy))
            wells[well_label] = well

        # Create a WellSample and associate the image
        ws = WellSampleI()
        ws.image = ImageI(img_id, False)
        ws.well = wells[well_label]
        ws.setPosX(omero.model.LengthI(lx, "MICROMETER"))
        ws.setPosY(omero.model.LengthI(ly, "MICROMETER"))
        ws.plateAcquisition = plate_acq
        wells[well_label].addWellSample(ws)

    # Save wells
    for well in wells.values():
        update_service.saveObject(well)

    # If enabled, delete dataset after processing
    delete_dataset = script_params.get("Delete_Dataset", False)
    delete_handle = None
    if delete_dataset and dataset_img_count == added_count:
        dcs = []
        options = None  # Don't delete images, only dataset
        dcs.append(omero.api.delete.DeleteCommand("/Dataset", dataset.id, options))
        delete_handle = conn.getDeleteService().queueDelete(dcs)

    return plate, link, "Plate created successfully.", delete_handle

def run_script():
    """
    Main script entry point, receives parameters and runs dataset conversion.
    """
    client = scripts.client(
        'Dataset_To_Plate_Dynamic.py',
        """Convert a Dataset of Images into a Plate dynamically based on
        image coordinates. Place images in wells based on X, Y positions.""",

        scripts.String("Data_Type", optional=False, values=[rstring('Dataset')], default="Dataset"),
        scripts.List("IDs", optional=False, description="List of Dataset IDs to process.").ofType(rlong(0)),
        scripts.String("First_Well", optional=False, default="A1", description="First well position (e.g., A1)."),
        scripts.String("Column_Names", optional=False, default='number', description="Plate column naming."),
        scripts.String("Row_Names", optional=False, default='letter', description="Plate row naming."),
        scripts.String("Screen", description="Optional: Enter ID of an existing screen."),

         version="4.3.2",
        authors=["William Moore", "OME Team"],
        institutions=["University of Dundee"],
        contact="ome-users@lists.openmicroscopy.org.uk",
    )

    try:
        script_params = client.getInputs(unwrap=True)
        conn = BlitzGateway(client_obj=client)

        screen = None
        if "Screen" in script_params and script_params["Screen"]:
            screen = conn.getObject("Screen", int(script_params["Screen"]))

        plates = []
        messages = []
        for dataset_id in script_params["IDs"]:
            plate, link, message = dataset_to_plate(conn, script_params, dataset_id, screen)
            if plate:
                plates.append(plate)
            messages.append(message)

        client.setOutput("Message", rstring(" ".join(messages)))
        if plates:
            client.setOutput("New_Object", robject(plates[0]))

    finally:
        client.closeSession()


if __name__ == "__main__":
    run_script()

# import omero.scripts as scripts
# from omero.gateway import BlitzGateway
# import omero

# from omero.rtypes import rint, rlong, rstring, robject, unwrap


# def add_images_to_plate(conn, images, plate_id, column, row, remove_from=None):
#     """
#     Add the Images to a Plate, creating a new well at the specified column and
#     row
#     NB - This will fail if there is already a well at that point
#     """
#     update_service = conn.getUpdateService()

#     well = omero.model.WellI()
#     well.plate = omero.model.PlateI(plate_id, False)
#     well.column = rint(column)
#     well.row = rint(row)

#     try:
#         for image in images:
#             ws = omero.model.WellSampleI()
#             ws.image = omero.model.ImageI(image.id, False)
#             ws.well = well
#             well.addWellSample(ws)
#         update_service.saveObject(well)
#     except Exception:
#         return False

#     # remove from Datast
#     for image in images:
#         if remove_from is not None:
#             links = list(image.getParentLinks(remove_from.id))
#             link_ids = [link.id for link in links]
#             conn.deleteObjects('DatasetImageLink', link_ids)
#     return True


# def dataset_to_plate(conn, script_params, dataset_id, screen):

#     dataset = conn.getObject("Dataset", dataset_id)
#     if dataset is None:
#         return

#     update_service = conn.getUpdateService()

#     # create Plate
#     plate = omero.model.PlateI()
#     plate.name = omero.rtypes.RStringI(dataset.name)
#     plate.columnNamingConvention = rstring(str(script_params["Column_Names"]))
#     # 'letter' or 'number'
#     plate.rowNamingConvention = rstring(str(script_params["Row_Names"]))
#     plate = update_service.saveAndReturnObject(plate)

#     if screen is not None and screen.canLink():
#         link = omero.model.ScreenPlateLinkI()
#         link.parent = omero.model.ScreenI(screen.id, False)
#         link.child = omero.model.PlateI(plate.id.val, False)
#         update_service.saveObject(link)
#     else:
#         link = None

#     row = 0
#     col = 0

#     first_axis_is_row = script_params["First_Axis"] == 'row'
#     axis_count = script_params["First_Axis_Count"]

#     # sort images by name
#     images = list(dataset.listChildren())
#     dataset_img_count = len(images)
#     if "Filter_Names" in script_params:
#         filter_by = script_params["Filter_Names"]
#         images = [i for i in images if i.getName().find(filter_by) >= 0]
#     images.sort(key=lambda x: x.name.lower())

#     # Do we try to remove images from Dataset and Delte Datset when/if empty?
#     remove_from = None
#     remove_dataset = "Remove_From_Dataset" in script_params and \
#         script_params["Remove_From_Dataset"]
#     if remove_dataset:
#         remove_from = dataset

#     images_per_well = script_params["Images_Per_Well"]
#     image_index = 0

#     while image_index < len(images):

#         well_images = images[image_index: image_index + images_per_well]
#         added_count = add_images_to_plate(conn, well_images,
#                                           plate.getId().getValue(),
#                                           col, row, remove_from)
#         image_index += images_per_well

#         # update row and column index
#         if first_axis_is_row:
#             row += 1
#             if row >= axis_count:
#                 row = 0
#                 col += 1
#         else:
#             col += 1
#             if col >= axis_count:
#                 col = 0
#                 row += 1

#     # if user wanted to delete dataset, AND it's empty we can delete dataset
#     delete_dataset = False   # Turning this functionality off for now.
#     delete_handle = None
#     if delete_dataset:
#         if dataset_img_count == added_count:
#             dcs = list()
#             options = None  # {'/Image': 'KEEP'}    # don't delete the images!
#             dcs.append(omero.api.delete.DeleteCommand(
#                 "/Dataset", dataset.id, options))
#             delete_handle = conn.getDeleteService().queueDelete(dcs)
#     return plate, link, delete_handle


# def datasets_to_plates(conn, script_params):

#     update_service = conn.getUpdateService()

#     message = ""

#     # Get the datasets ID
#     dtype = script_params['Data_Type']
#     ids = script_params['IDs']
#     datasets = list(conn.getObjects(dtype, ids))

#     def has_images_linked_to_well(dataset):
#         params = omero.sys.ParametersI()
#         query = "select count(well) from Well as well "\
#                 "left outer join well.wellSamples as ws " \
#                 "left outer join ws.image as img "\
#                 "where img.id in (:ids)"
#         params.addIds([i.getId() for i in dataset.listChildren()])
#         n_wells = unwrap(conn.getQueryService().projection(
#             query, params, conn.SERVICE_OPTS)[0])[0]
#         if n_wells > 0:
#             return True
#         else:
#             return False

#     # Exclude datasets containing images already linked to a well
#     n_datasets = len(datasets)
#     datasets = [x for x in datasets if not has_images_linked_to_well(x)]
#     if len(datasets) < n_datasets:
#         message += "Excluded %s out of %s dataset(s). " \
#             % (n_datasets - len(datasets), n_datasets)

#     # Return if all input dataset are not found or excluded
#     if not datasets:
#         return None, message

#     # Filter dataset IDs by permissions
#     ids = [ds.getId() for ds in datasets if ds.canLink()]
#     if len(ids) != len(datasets):
#         perm_ids = [str(ds.getId()) for ds in datasets if not ds.canLink()]
#         message += "You do not have the permissions to add the images from"\
#             " the dataset(s): %s." % ",".join(perm_ids)
#     if not ids:
#         return None, message

#     # find or create Screen if specified
#     screen = None
#     newscreen = None
#     if "Screen" in script_params and len(script_params["Screen"]) > 0:
#         s = script_params["Screen"]
#         # see if this is ID of existing screen
#         try:
#             screen_id = int(s)
#             screen = conn.getObject("Screen", screen_id)
#         except ValueError:
#             pass
#         # if not, create one
#         if screen is None:
#             newscreen = omero.model.ScreenI()
#             newscreen.name = rstring(s)
#             newscreen = update_service.saveAndReturnObject(newscreen)
#             screen = conn.getObject("Screen", newscreen.getId().getValue())

#     plates = []
#     links = []
#     deletes = []
#     for dataset_id in ids:
#         plate, link, delete_handle = dataset_to_plate(conn, script_params,
#                                                       dataset_id, screen)
#         if plate is not None:
#             plates.append(plate)
#         if link is not None:
#             links.append(link)
#         if delete_handle is not None:
#             deletes.append(delete_handle)

#     # wait for any deletes to finish
#     for handle in deletes:
#         cb = omero.callbacks.DeleteCallbackI(conn.c, handle)
#         while True:  # ms
#             if cb.block(100) is not None:
#                 break

#     if newscreen:
#         message += "New screen created: %s." % newscreen.getName().getValue()
#         robj = newscreen
#     elif plates:
#         robj = plates[0]
#     else:
#         robj = None

#     if plates:
#         if len(plates) == 1:
#             plate = plates[0]
#             message += " New plate created: %s" % plate.getName().getValue()
#         else:
#             message += " %s plates created" % len(plates)
#         if len(plates) == len(links):
#             message += "."
#         else:
#             message += " but could not be attached."
#     else:
#         message += "No plate created."
#     return robj, message


# def run_script():
#     """
#     The main entry point of the script, as called by the client via the
#     scripting service, passing the required parameters.
#     """

#     data_types = [rstring('Dataset')]
#     first_axis = [rstring('column'), rstring('row')]
#     row_col_naming = [rstring('letter'), rstring('number')]

#     client = scripts.client(
#         'Dataset_To_Plate.py',
#         """Take a Dataset of Images and put them in a new Plate, \
# arranging them into rows or columns as desired.
# Optionally add the Plate to a new or existing Screen.
# See http://help.openmicroscopy.org/scripts.html""",

#         scripts.String(
#             "Data_Type", optional=False, grouping="1",
#             description="Choose source of images (only Dataset supported)",
#             values=data_types, default="Dataset"),

#         scripts.List(
#             "IDs", optional=False, grouping="2",
#             description="List of Dataset IDs to convert to new"
#             " Plates.").ofType(rlong(0)),

#         scripts.String(
#             "Filter_Names", grouping="2.1",
#             description="Filter the images by names that contain this value"),

#         scripts.String(
#             "First_Axis", grouping="3", optional=False, default='column',
#             values=first_axis,
#             description="""Arrange images accross 'column' first or down"
#             " 'row'"""),

#         scripts.Int(
#             "First_Axis_Count", grouping="3.1", optional=False, default=12,
#             description="Number of Rows or Columns in the 'First Axis'",
#             min=1),

#         scripts.Int(
#             "Images_Per_Well", grouping="3.2", optional=False, default=1,
#             description="Number of Images (Well Samples) per Well",
#             min=1),

#         scripts.String(
#             "Column_Names", grouping="4", optional=False, default='number',
#             values=row_col_naming,
#             description="""Name plate columns with 'number' or 'letter'"""),

#         scripts.String(
#             "Row_Names", grouping="5", optional=False, default='letter',
#             values=row_col_naming,
#             description="""Name plate rows with 'number' or 'letter'"""),

#         scripts.String(
#             "Screen", grouping="6",
#             description="Option: put Plate(s) in a Screen. Enter Name of new"
#             " screen or ID of existing screen"""),

#         scripts.Bool(
#             "Remove_From_Dataset", grouping="7", default=True,
#             description="Remove Images from Dataset as they are added to"
#             " Plate"),

#         version="4.3.2",
#         authors=["William Moore", "OME Team"],
#         institutions=["University of Dundee"],
#         contact="ome-users@lists.openmicroscopy.org.uk",
#     )

#     try:
#         script_params = client.getInputs(unwrap=True)

#         # wrap client to use the Blitz Gateway
#         conn = BlitzGateway(client_obj=client)

#         # convert Dataset(s) to Plate(s). Returns new plates or screen
#         new_obj, message = datasets_to_plates(conn, script_params)

#         client.setOutput("Message", rstring(message))
#         if new_obj:
#             client.setOutput("New_Object", robject(new_obj))

#     finally:
#         client.closeSession()


# if __name__ == "__main__":
#     run_script()
