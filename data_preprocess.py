import os
from functools import partial
from tqdm import tqdm
import numpy as np

from osgeo import gdal
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box, Polygon, MultiPoint, MultiPolygon


os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2


def get_polygon_coordinates(polygon, separate_ex_in=False):
    """
    Get all coordinates of a Polygon object.
    """
    exterior = list(polygon.exterior.coords)
    # when a Polygon has hole(s), its interiors is not empty
    interiors = []
    for ring in polygon.interiors:  # interiors is a list of LineRing
        # for LineRing, coordinate values are accessed via the coords property:
        interiors += list(ring.coords)

    # Note that the priority relationship between comma and if-else, the bracket is necessary
    # If not, the return value will be (exterior, exterior + interiors) given separate_ex_in is False
    return (exterior, interiors) if separate_ex_in else exterior + interiors


def geo_coords_to_pixel_coords(geo_coords, geo_upper_left, geo_h_w, pix_h_w):
    """
    Convert coordinates under geographic coordinate system into
    coordinates under pixel coordinate system of given image.

    Geographic coordinate system:
         ↑ y
        o------→ x
        The bottom left point is NOT necessarily the origin.
    Pixel coordinate system:
        o------→ x
         ↓ y
        The upper left point is just the origin.

    @param geo_coords: a list of coordinates under geographic coordinate system
    @param geo_upper_left: the coordinate of upper left point in under geographic coordinate system
    @param geo_h_w: the height and width of the image under geographic coordinate system
    @param pix_h_w: the height and width of the image under pixel coordinate system

    @return a list contains coordinates under pixel coordinate system
    """
    pixel_coords = []

    for geo_x, geo_y in geo_coords:
        pixel_x = (geo_x - geo_upper_left[0]) * (pix_h_w[1] / geo_h_w[1])
        pixel_y = (geo_upper_left[1] - geo_y) * (pix_h_w[0] / geo_h_w[0])
        pixel_coords.append([pixel_x, pixel_y])

    return pixel_coords


def pixel_coords_to_geo_coords(pixel_coords, geo_upper_left, geo_h_w, pix_h_w):
    """
    The reverse operation of function `geo_coords_to_pixel_coords`.
    """
    geo_coords = []
    for pixel_x, pixel_y in pixel_coords:
        geo_x = geo_upper_left[0] + pixel_x * (geo_h_w[1] / pix_h_w[1])
        geo_y = geo_upper_left[1] - pixel_y * (geo_h_w[0] / pix_h_w[0])
        geo_coords.append([geo_x, geo_y])

    return geo_coords


def pixel_coords_to_patch_coords(pixel_coords, patch_upper_left_point):
    """
    Convert coordinates in the large image under pixel coordinate system into coordinates in the patch.
    """
    return [[x - patch_upper_left_point[0], y - patch_upper_left_point[1]] for x, y in pixel_coords]


def read_tif_with_info(tif_path):
    tif = gdal.Open(tif_path)

    geo_transform = tif.GetGeoTransform()  # get the affine matrix
    min_x, max_y = geo_transform[0], geo_transform[3]
    resolution_horizon, resolution_vertical = geo_transform[1], abs(geo_transform[5])
    rotation_angle_x, rotation_angle_y = geo_transform[2], abs(geo_transform[4])
    assert rotation_angle_x == 0 and rotation_angle_y == 0, 'rotation angle is not zero'

    pixel_count_x, pixel_count_y = tif.RasterXSize, tif.RasterYSize
    max_x = min_x + pixel_count_x * resolution_horizon
    min_y = max_y - pixel_count_y * resolution_vertical

    geo_h_w = (max_y - min_y, max_x - min_x)  # the height and width under geographic coordinate system
    geo_upper_left_point = (min_x, max_y)  # the coordinate of the upper left point under geographic coordinate system
    pix_h_w = (pixel_count_y, pixel_count_x)  # the height and width under pixel coordinate system

    return tif, geo_h_w, geo_upper_left_point, pix_h_w


def generate_mask_from_shp_for_patch(union_roads, patch_h_w, patch_upper_left_point,
                                     geo_upper_left_point, geo_h_w, pix_h_w,
                                     output_equivalent_rect=False, save_dir='', patch_id=1):
    """
    Generate the corresponding mask from shapefile for a patch.

    @param union_roads: the annotation data, a MultiPolygon object
    @param patch_h_w: height and width of the patch
    @param patch_upper_left_point: coordinate of the upper left point of the patch, in pixel coordinates.
    @param geo_upper_left_point: coordinate of the upper left point of the tif image, in geo coordinates
    @param geo_h_w: height and width of the tif image, in geo coordinates
    @param pix_h_w: height and width of the tif image, in pixel coordinates
    @param output_equivalent_rect: whether output shapefiles to visualize equivalent rectangles
    @param save_dir: directory to save equivalent rectangle shapefiles, only necessary when setting
            `output_equivalent_rect` as True
    @param patch_id: a value to identify the current patch, used to constitute part of the filename
            of equivalent rectangle shapefile. Only necessary when setting `output_equivalent_rect` as True

    @return: the mask (NumPy ndarray)
    """

    # coordinates of the bottom left point and upper right point of the patch in pixel coordinates
    patch_bottom_left_upper_right = [[patch_upper_left_point[0], patch_upper_left_point[1] + patch_h_w[0]],
                                     [patch_upper_left_point[0] + patch_h_w[1], patch_upper_left_point[1]]]

    # convert the patch's pixel coordinates into geographic coordinates
    equivalent_rect_coords = pixel_coords_to_geo_coords(patch_bottom_left_upper_right,
                                                        geo_upper_left=geo_upper_left_point,
                                                        geo_h_w=geo_h_w,
                                                        pix_h_w=pix_h_w)
    # create a polygon that equivalent to the patch
    equivalent_rect = box(minx=equivalent_rect_coords[0][0],
                          miny=equivalent_rect_coords[0][1],
                          maxx=equivalent_rect_coords[1][0],
                          maxy=equivalent_rect_coords[1][1])

    if output_equivalent_rect:
        assert save_dir != '', 'save_dir can not be empty when setting output_equivalent_rect as True!'
        save_geometry_object_into_shp(equivalent_rect,
                                      save_path=os.path.join(save_dir, f'{patch_id}.shp'),
                                      crs=union_roads.crs)

    geo2pix_func = partial(geo_coords_to_pixel_coords,
                           geo_upper_left=geo_upper_left_point,
                           geo_h_w=geo_h_w,
                           pix_h_w=pix_h_w)

    # calculate the intersection set between the patch polygon and the annotations
    intersections = []

    def transform_intersection_polygon(intersection):
        pixel_coords = geo2pix_func(geo_coords=get_polygon_coordinates(intersection))
        patch_pixel_coords = pixel_coords_to_patch_coords(pixel_coords, patch_upper_left_point)
        return np.array(patch_pixel_coords, dtype=np.int32)

    for road in union_roads.geoms:
        inter = equivalent_rect.intersection(road)
        if not inter.is_empty:
            if type(inter) == MultiPolygon:
                for p in inter.geoms:
                    intersections.append(transform_intersection_polygon(p))
            elif type(inter) == Polygon:
                intersections.append(transform_intersection_polygon(inter))

    mask = np.zeros((patch_h_w[0], patch_h_w[1], 1), dtype=np.uint8)  # create a corresponding mask image
    if len(intersections) != 0:
        cv2.fillPoly(img=mask, pts=np.array(intersections), color=255)
        # The coordinates of a polygon with hole(s) are divided into exterior & interiors and are
        # concatenated into one polygon coordinate array.
        # cv2.fillPoly regards them as the same closed polygon.
        # (Doubt: How to indicate an area with holes in cv2.fillPoly?)
        # This case will result in there exists a thin line connecting the first point and the last
        # point of the coordinate array.
        # A dirty method to solve this: draw a black line to override the original line
        for inter in intersections:
            if (inter[0] != inter[-1]).all():
                cv2.line(img=mask, pt1=tuple(inter[0].tolist()), pt2=tuple(inter[-1].tolist()),
                         color=0, thickness=1)

    return mask


def cut_large_image_into_patches(tif_path, patch_h_w, patch_ext,
                                 save_dir_with_ann, save_dir_without_ann,
                                 ann_src='shp', shp_path='', mask_path=''):
    """
    Cut a large tif image with its annotations into paired patches.

    @param tif_path: file path of the tif image
    @param patch_h_w: height and width of the patches
    @param patch_ext: file extension of the patches
    @param save_dir_with_ann: directory to save patches with annotations
    @param save_dir_without_ann: directory to save patches without annotations
    @param ann_src: source of annotation, "shp" or "mask"
    @param shp_path: file path of shapefile, necessary when setting ann_src as "shp"
    @param mask_path: file path of mask image, necessary when setting ann_src as "mask"
    """

    assert ann_src in ('shp', 'mask'), 'ann_src must be chosen from "shp" and "mask"!'

    tif, geo_h_w, geo_upper_left_point, pix_h_w = read_tif_with_info(tif_path)
    pix_height, pix_width = pix_h_w

    if ann_src == 'shp':
        assert shp_path != '', 'param shp_path can not be empty if set ann_src as "shp"!'
        roads = gpd.read_file(shp_path)['geometry']  # Read annotations from shapefile, roads is a GeoSeries object
        union_roads = unary_union([road for road in roads if road is not None])
        mask_patch_generator = partial(generate_mask_from_shp_for_patch, union_roads=union_roads,
                                       geo_upper_left_point=geo_upper_left_point,
                                       geo_h_w=geo_h_w, pix_h_w=pix_h_w)
    else:
        assert mask_path != '', 'param mask_path can not be empty if set ann_src as "mask"!'
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # mask has two dims

        assert pix_h_w == mask.shape[:2], 'Tif and mask have different sizes!'

        def mask_patch_generator(patch_upper_left_point, patch_h_w):
            # add a dimension as the third
            return np.expand_dims(mask[patch_upper_left_point[1]: patch_upper_left_point[1] + patch_h_w[0],
                                  patch_upper_left_point[0]: patch_upper_left_point[0] + patch_h_w[1]], axis=2)

    cnt = 0
    for i in tqdm(range(0, pix_height, patch_h_w[0])):
        for j in range(0, pix_width, patch_h_w[1]):
            actual_patch_height = min(patch_h_w[0], pix_height - i)
            actual_patch_width = min(patch_h_w[1], pix_width - j)
            img_patch = tif.ReadAsArray(j, i, actual_patch_width, actual_patch_height,
                                        interleave='pixel')  # set interleave as 'pixel' to make the color channels last

            if not np.any(img_patch):  # if there are just all zeros in img_patch, skip
                continue

            mask_patch = mask_patch_generator(patch_upper_left_point=(j, i),
                                              patch_h_w=(actual_patch_height, actual_patch_width))

            # if all elements of mask_patch is zero, it means there is no annotation for the current img_patch or
            # there is no road in the current patch
            # img_patch_filename = img_filename.replace(os.path.splitext(img_filename)[1], f'_{cnt}{patch_ext}')
            # mask_patch_filename = img_patch_filename.replace(patch_ext, '_mask.png')
            img_patch_filename = f'{cnt}{patch_ext}'
            mask_patch_filename = img_patch_filename.replace(patch_ext, '_mask.png')
            img_patch_save_path = os.path.join(save_dir_with_ann, img_patch_filename) \
                if np.any(mask_patch) else os.path.join(save_dir_without_ann, img_patch_filename)
            mask_patch_save_path = os.path.join(save_dir_with_ann, mask_patch_filename) \
                if np.any(mask_patch) else os.path.join(save_dir_without_ann, mask_patch_filename)

            # if the actual patch size is smaller than setting, pad them to the given size with 0
            if actual_patch_height != patch_h_w[0] or actual_patch_width != patch_h_w[1]:
                temp_img_patch = np.zeros((patch_h_w[0], patch_h_w[1], tif.RasterCount), dtype=img_patch.dtype)
                temp_img_patch[:actual_patch_height, :actual_patch_width] = img_patch

                temp_mask_patch = np.zeros((patch_h_w[0], patch_h_w[1], 1), dtype=np.uint8)
                temp_mask_patch[:actual_patch_height, :actual_patch_width] = mask_patch
                img_patch, mask_patch = temp_img_patch, temp_mask_patch

            cv2.imencode(patch_ext, img_patch)[1].tofile(img_patch_save_path)
            cv2.imencode('.png', mask_patch)[1].tofile(mask_patch_save_path)

            cnt += 1

    print(f'Output {cnt} pairs of image and mask in total.')


def polygon_to_points(shapefile_or_polygon, save_path, crs=None):
    """
    Convert a shapefile (its geometry is Polygon (MultiPolygon))
    or a Polygon object into a MultiPoint object, and save it to another shapefile.

    @param shapefile_or_polygon: a str/Polygon object
    @param save_path:
    @param crs: necessary when passing into a Polygon object
    """
    # Case 1. shp_path_or_polygon is a Polygon object
    if isinstance(shapefile_or_polygon, Polygon):
        assert crs is not None, 'crs is unknown'
        mp = MultiPoint(points=get_polygon_coordinates(shapefile_or_polygon))
        gpd.GeoSeries(mp, crs=crs).to_file(save_path, driver='ESRI Shapefile', encoding='utf-8')
        return

    # Case 2. shp_path_or_polygon is a str indicating the path of a shapefile
    data = gpd.read_file(shapefile_or_polygon)
    roads = data['geometry']
    points = []
    for road in roads:
        if type(road) == MultiPolygon:
            for poly in road.geoms:
                points.extend(get_polygon_coordinates(poly))
        elif type(road) == Polygon:
            points.extend(get_polygon_coordinates(road))
        elif road is None:
            continue
        else:
            raise TypeError(f'Unsupported type: {type(road)}')

    mp = MultiPoint(points=points)
    save_geometry_object_into_shp(mp, save_path, data.crs)


def save_geometry_object_into_shp(geometry_obj, save_path, crs):
    """
    Save a geometry object (e.g. Polygon, MultiPoint) into a shapefile.
    """
    gpd.GeoSeries(geometry_obj, crs=crs).to_file(save_path, driver='ESRI Shapefile', encoding='utf-8')


def convert_shp_to_single_mask(tif_path, shp_path, save_path, block_h_w=(256, 256)):
    """
    Convert a shapefile containing annotations to a single huge mask image, block by block.

    @param tif_path:
    @param shp_path:
    @param save_path:
    @param block_h_w:
    """
    _, geo_h_w, geo_upper_left_point, pix_h_w = read_tif_with_info(tif_path)
    pix_height, pix_width = pix_h_w

    # roads = gpd.read_file(shp_path)['geometry']
    shp = gpd.read_file(shp_path)
    # https://stackoverflow.com/questions/63955752/topologicalerror-the-operation-geosintersection-r-could-not-be-performed
    shp['geometry'] = shp.buffer(0)
    roads = shp['geometry']
    union_roads = unary_union([road for road in roads if road is not None])     # merge geometry objects
    mask_patch_generator = partial(generate_mask_from_shp_for_patch, union_roads=union_roads,
                                   geo_upper_left_point=geo_upper_left_point,
                                   geo_h_w=geo_h_w, pix_h_w=pix_h_w)

    # create a mask image with same size
    res_mask = np.zeros((pix_h_w[0], pix_h_w[1], 1), dtype=np.uint8)

    for i in tqdm(range(0, pix_height, block_h_w[0])):
        for j in range(0, pix_width, block_h_w[1]):
            actual_block_height = min(block_h_w[0], pix_height - i)
            actual_block_width = min(block_h_w[1], pix_width - j)

            mask_patch = mask_patch_generator(patch_upper_left_point=(j, i),
                                              patch_h_w=(actual_block_height, actual_block_width))
            res_mask[i:i + actual_block_height, j:j + actual_block_width] = mask_patch

    cv2.imencode('.png', res_mask)[1].tofile(save_path)


def convert_shp_to_single_mask_simple_polygons(tif_path, shp_path, save_path):
    _, geo_h_w, geo_upper_left_point, pix_h_w = read_tif_with_info(tif_path)
    pix_height, pix_width = pix_h_w

    polygons = gpd.read_file(shp_path)['geometry']
    res_mask = np.zeros((pix_height, pix_width, 1), dtype=np.uint8)

    pix_polygons = []
    for poly in tqdm(polygons):
        if type(poly) == MultiPolygon:
            for p in poly.geoms:
                pix_poly = geo_coords_to_pixel_coords(list(p.exterior.coords),
                                                      geo_upper_left=geo_upper_left_point, geo_h_w=geo_h_w,
                                                      pix_h_w=pix_h_w)
                pix_polygons.append(np.array(pix_poly, dtype=np.int32))
        else:
            pix_poly = geo_coords_to_pixel_coords(list(poly.exterior.coords),
                                                  geo_upper_left=geo_upper_left_point, geo_h_w=geo_h_w, pix_h_w=pix_h_w)
            pix_polygons.append(np.array(pix_poly, dtype=np.int32))

    cv2.fillPoly(img=res_mask, pts=np.array(pix_polygons), color=255)
    cv2.imencode('.png', res_mask)[1].tofile(save_path)


def correct_shp_crs(shp_path, target_crs, save_path):
    """
    Correct the given shapefile's crs to target value.
    Equivalent to QGIS: Processing -> Toolbox -> Assign projection

    @param shp_path:
    @param target_crs: a str like 'epsg:32649'
    @param save_path:
    """
    data = gpd.read_file(shp_path)
    data.to_crs(crs=target_crs, epsg=int(target_crs.split(':')[1]), inplace=True)
    data.to_file(save_path, driver='ESRI Shapefile', encoding='utf-8')


def decompose_shp(shp_path, save_dir):
    """
    Decompose the geometry elements in the given shapefile into separate shapefiles.
    """
    data = gpd.read_file(shp_path)
    geometry = data['geometry']
    geometry = unary_union([geom for geom in geometry if geom is not None])

    cnt = 0
    for geom in geometry:
        cnt += 1
        save_geometry_object_into_shp(geom, os.path.join(save_dir, f'{cnt}.shp'), data.crs)


def main():
    # convert_shp_to_single_mask(tif_path=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\intermediate_data"
    #                                     r"\FenHe\FenHe_Merged_ArcGIS.tif",
    #                            shp_path=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\汾河道路\汾河.shp",
    #                            save_path='./mask_b50x50.png', block_h_w=(50, 50))

    # cut_large_image_into_patches(tif_path=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\intermediate_data"
    #                                       r"\FenHe\FenHe_Merged_ArcGIS.tif",
    #                              patch_h_w=(1024, 1024),
    #                              patch_ext='.jpg',
    #                              save_dir_with_ann=r'F:\MachineLearning-Datasets\20211105-减灾数据-标注'
    #                                                r'\dataset\FenHe_1024x1024_png\with_ann',
    #                              save_dir_without_ann=r'F:\MachineLearning-Datasets\20211105-减灾数据-标注'
    #                                                   r'\dataset\FenHe_1024x1024_png\without_ann',
    #                              ann_src='mask',
    #                              # shp_path=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\汾河道路\汾河.shp",
    #                              mask_path=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\intermediate_data"
    #                                        r"\FenHe\shp_to_mask\Block_by_Block\mask_b50x50_update.png")

    # convert_shp_to_single_mask_simple_polygons(tif_path=r"F:\MachineLearning-Datasets\road_damage\舟曲道路损毁标注示例\DOM影像\DOM.tif",
    #                                            shp_path=r"F:\MachineLearning-Datasets\road_damage\舟曲道路损毁标注示例\标注示例\Road_damage.shp",
    #                                            save_path='ZhouQu.png')

    cut_large_image_into_patches(tif_path=r"I:\JieXiu\JieXiu_Merged.tif",
                                 patch_h_w=(1024, 1024),
                                 patch_ext='.jpg',
                                 save_dir_with_ann=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\dataset\temp\with_ann",
                                 save_dir_without_ann=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\dataset\temp\without_ann",
                                 ann_src='mask',
                                 mask_path=r"F:\MachineLearning-Datasets\20211105-减灾数据-标注\intermediate_data\JieXiu\mask_b50x50_JieXiu.png")


if __name__ == '__main__':
    main()
