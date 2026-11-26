// STEP 2 - Baseline flood map (UN-SPIDER change-detection method). Karnataka 2019 by default;
// for Kerala use: aoi [76.0, 9.0, 76.8, 10.5], BEFORE ['2018-05-01','2018-05-31'], AFTER ['2018-08-14','2018-08-24'], name 'KL2018'.
var aoi = ee.Geometry.Rectangle([74.5, 15.5, 76.0, 17.0]);
var BEFORE = ['2019-07-10', '2019-07-31'], AFTER = ['2019-08-06', '2019-08-14'], NAME = 'KA2019';

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select('VH');
var before = s1.filterDate(BEFORE[0], BEFORE[1]).median().focal_median(50, 'circle', 'meters').clip(aoi);
var after  = s1.filterDate(AFTER[0],  AFTER[1]).mosaic().focal_median(50, 'circle', 'meters').clip(aoi);

var flood = after.divide(before).gt(1.25);                                    // pixel got ~25% darker
flood = flood.where(ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality').gte(10), 0);  // not rivers/lakes
flood = flood.where(ee.Terrain.slope(ee.Image('WWF/HydroSHEDS/03VFDEM')).gt(5), 0);              // not hills
flood = flood.selfMask();
flood = flood.updateMask(flood.connectedPixelCount(25, true).gte(8));         // no specks

Map.centerObject(aoi, 9);
Map.addLayer(before, {min: -25, max: -5}, 'Before (VH)');
Map.addLayer(after,  {min: -25, max: -5}, 'After (VH)');
Map.addLayer(flood,  {palette: ['0055ff']}, 'Flood');

var wc = ee.ImageCollection('ESA/WorldCover/v200').first();
function km2(mask) {
  return ee.Number(mask.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, scale: 30, maxPixels: 1e10}).values().get(0)).divide(1e6);
}
print('Flooded area (km2):',     km2(flood));
print('Flooded cropland (km2):', km2(flood.updateMask(wc.eq(40))));
print('Flooded built-up (km2):', km2(flood.updateMask(wc.eq(50))));
Export.image.toDrive({image: flood.unmask(0).toByte(), description: NAME + '_flood_baseline',
                      region: aoi, scale: 30, crs: 'EPSG:4326', maxPixels: 1e10});
