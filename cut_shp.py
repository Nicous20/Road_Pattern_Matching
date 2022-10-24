from osgeo import gdal, ogr
import os


def cut(shp_path, lonlat, out_shp, geom_type=ogr.wkbMultiLineString):
    data = ogr.Open(shp_path, gdal.GA_ReadOnly)
    layer = data.GetLayer()
    mpg = ogr.Geometry(ogr.wkbMultiPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(lonlat[0],lonlat[2])
    ring.AddPoint_2D(lonlat[1],lonlat[2])
    ring.AddPoint_2D(lonlat[1],lonlat[3])
    ring.AddPoint_2D(lonlat[0],lonlat[3])
    ring.CloseRings()
    py = ogr.Geometry(ogr.wkbPolygon)
    py.AddGeometry(ring)
    mpg.AddGeometry(py)

    #uid = uuid.uuid4() #随机文件名，防止并行重复，用完即删
    dst_filename = out_shp
    drv = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(dst_filename): #若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(dst_filename)
    dst_ds = drv.CreateDataSource(dst_filename)
    dst_layername = 'out'
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=layer.GetSpatialRef(), geom_type = geom_type)

    for i,fea in enumerate(layer):
        poly = fea.GetGeometryRef().Intersection(mpg)
        if poly:
            feature = ogr.Feature(dst_layer.GetLayerDefn())
            feature.SetGeometry(poly.Clone())
            dst_layer.CreateFeature(feature)
    data.Destroy()
    dst_ds.Destroy()

# 最小纬度，最大纬度   最小经度  最大经度
lonlat = (110.149639, 110.155234, 34.158117, 34.164735)
cut(r"D:\Road_code\dataset\2020陕西商洛山洪\灾后提取结果\洛南灾后道路提取结果\洛南\centerline_洛南灾区.shp", lonlat,
    r"D:\Road_code\SuperRetina\tools\cut_shps")
cut(r"D:\Road_code\dataset\2020陕西商洛山洪\灾前提取结果\centerline.shp", lonlat,
    r"D:\Road_code\SuperRetina\tools\cut_shps_after")
