// STEP 3 - Flood frequency: how many monsoons (2016-2024) each pixel flooded. Used to price insurance.
// If whole hillsides light up red, change -22 to -24.
var regions = {KA: ee.Geometry.Rectangle([74.5, 15.5, 76.0, 17.0]), KL: ee.Geometry.Rectangle([76.0, 9.0, 76.8, 10.5])};
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select('VH');
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality');
var slope = ee.Terrain.slope(ee.Image('WWF/HydroSHEDS/03VFDEM'));

Object.keys(regions).forEach(function (name) {
  var aoi = regions[name];
  var years = ee.List.sequence(2016, 2024).map(function (y) {
    y = ee.Number(y);
    var dry = s1.filterBounds(aoi).filterDate(ee.Date.fromYMD(y, 2, 1), ee.Date.fromYMD(y, 4, 30))
                .median().focal_median(50, 'circle', 'meters');
    var wet = s1.filterBounds(aoi).filterDate(ee.Date.fromYMD(y, 6, 1), ee.Date.fromYMD(y, 10, 31))
                .min().focal_median(50, 'circle', 'meters');
    return wet.lt(-22).and(dry.gt(-18)).rename('flooded');     // water in monsoon, land in dry season
  });
  var freq = ee.ImageCollection.fromImages(years).sum().where(gsw.gte(10), 0).where(slope.gt(5), 0).clip(aoi);
  Map.addLayer(freq.selfMask(), {min: 1, max: 9, palette: ['ffffb2', 'fd8d3c', 'bd0026']}, name + ' flood years');
  Export.image.toDrive({image: freq.toByte(), description: name + '_flood_frequency_2016_2024',
                        region: aoi, scale: 30, crs: 'EPSG:4326', maxPixels: 1e10});
});
Map.setCenter(75.5, 13.5, 7);
