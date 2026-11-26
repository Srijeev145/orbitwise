// STEP 1 - Export before/after Sentinel-1 radar images for both floods.
// Paste into code.earthengine.google.com, click Run, then Run each task in the Tasks tab.
// Check the printed dates: if an "after" list is empty, widen the dates or switch DESCENDING -> ASCENDING
// (keep before and after on the same direction).
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH']);

function exportPair(name, aoi, beforeDates, afterDates) {
  var col = s1.filterBounds(aoi);
  print(name + ' before dates', col.filterDate(beforeDates[0], beforeDates[1]).aggregate_array('system:index'));
  print(name + ' after dates',  col.filterDate(afterDates[0],  afterDates[1]).aggregate_array('system:index'));
  var before = col.filterDate(beforeDates[0], beforeDates[1]).median().clip(aoi);
  var after  = col.filterDate(afterDates[0],  afterDates[1]).mosaic().clip(aoi);
  [['before', before], ['after', after]].forEach(function (p) {
    Export.image.toDrive({image: p[1].multiply(100).toInt16(), description: name + '_' + p[0],
                          region: aoi, scale: 50, crs: 'EPSG:4326', maxPixels: 1e10});
  });
}
exportPair('KA2019', ee.Geometry.Rectangle([74.5, 15.5, 76.0, 17.0]), ['2019-07-10', '2019-07-31'], ['2019-08-06', '2019-08-14']);
exportPair('KL2018', ee.Geometry.Rectangle([76.0, 9.0, 76.8, 10.5]),  ['2018-05-01', '2018-05-31'], ['2018-08-14', '2018-08-24']);
