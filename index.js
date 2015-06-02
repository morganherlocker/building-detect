var getPixels = require('get-pixels')
var savePixels = require('save-pixels')
var ndarray = require('ndarray')
var zeros = require('zeros')
var fs = require('fs')
var brain = require('brain')
var queue = require('queue-async')
var unpack = require('ndarray-unpack')
var kmeans = require("clusterfck").kmeans

console.log('<link rel="stylesheet" type="text/css" href="style.css">')

var positives = fs.readdirSync(__dirname+'/positive').map(function(file){
  return __dirname+'/positive/'+file
})
var negatives = fs.readdirSync(__dirname+'/negative').map(function(file){
  return __dirname+'/negative/'+file
})
var tests = fs.readdirSync(__dirname+'/tests').map(function(file){
  return __dirname+'/tests/'+file
})

var net = new brain.NeuralNetwork()
var q = queue(1)
var q2 = queue(1)
var cases = []

positives.forEach(queueImage)
negatives.forEach(queueImage)

q.awaitAll(function(err, data){
  net.train(data, {
    //log: true,
    learningRate: 0.2,
    iterations: 1000000
  });
  
  tests.forEach(queueTest)
  q2.awaitAll(function(err, testData){
    testData.forEach(function(test){
      var building = net.run(test.input).building
      var classification = 'no'
      if(building >= .5) classification = 'yes'
      console.log('<h1>'+Math.round(building * 100000000) / 1000000+'</h1>')
      console.log('<div><img class="'+classification+'" src="tests/'+test.path.split('/')[test.path.split('/').length-1]+'"></div>')
      console.log('<div class="swatch-container">')
      test.clusters.forEach(function(cluster){
        console.log('<div class="swatch" style="background-color:rgb('+cluster.slice(0,3).join(',')+');"></div>')
      })
      console.log('</div>')
    })
  })
})

function queueImage(path){
  var building = 0
  if(path.split('/').indexOf('positive') !== -1) building = 1

  q.defer(function(path, done){
    getPixels(path, function(err, pixels){
      var heuristics = getHeuristics(pixels)

      var img = {
        input: heuristics.input,
        output: {building: building}
      }
      done(null, img)
    })
  }, path)
}

function queueTest(path){
  var building = 0
  if(path.split('/').indexOf('positive') !== -1) building = 1

  q2.defer(function(path, done){
    getPixels(path, function(err, pixels){
      var heuristics = getHeuristics(pixels)

      var img = {
        input: heuristics.input,
        output: {building: building},
        path: path,
        clusters: heuristics.clusters
      }
      done(null, img)
    })
  }, path)
}

function getHeuristics(pixels){
  var avgpx = avg(pixels)
  var lowpx = low(pixels)
  var highpx = high(pixels)
  var clusters = clusterPixels(pixels)

  var flatClusters = []
  clusters.forEach(function(cluster){
    flatClusters = flatClusters.concat(cluster)
  })

  var input = avgpx.concat(lowpx)
    .concat(highpx)
    .concat(flatClusters)
    .map(function(num){ return num / 255})

  return {
    input: input,
    avgpx: avgpx,
    lowpx: lowpx,
    highpx: highpx,
    clusters: clusters
  }
}

function avg (pixels) {
  var shape = pixels.shape.slice();
  var total = shape[0] * shape[1]
  var r = 0;
  var g = 0;
  var b = 0;
  for(var x = 0; x < shape[0]; x++) {
    for(var y = 0; y < shape[1]; y++) {
      r += pixels.get(x,y,0);
      g += pixels.get(x,y,1);
      b += pixels.get(x,y,2);
    }
  }
  return [r/total, g/total, b/total].map(Math.round);
}

function low (pixels) {
  var shape = pixels.shape.slice();
  var r = Infinity;
  var g = Infinity;
  var b = Infinity;
  for(var x = 0; x < shape[0]; x++) {
    for(var y = 0; y < shape[1]; y++) {
      if(r > pixels.get(x,y,0)) r = pixels.get(x,y,0)
      if(g > pixels.get(x,y,1)) g = pixels.get(x,y,1)
      if(b > pixels.get(x,y,2)) b = pixels.get(x,y,2)
    }
  }
  return [r,g,b];
}

function high (pixels) {
  var shape = pixels.shape.slice();
  var r = -Infinity;
  var g = -Infinity;
  var b = -Infinity;
  for(var x = 0; x < shape[0]; x++) {
    for(var y = 0; y < shape[1]; y++) {
      if(r < pixels.get(x,y,0)) r = pixels.get(x,y,0)
      if(g < pixels.get(x,y,1)) g = pixels.get(x,y,1)
      if(b < pixels.get(x,y,2)) b = pixels.get(x,y,2)
    }
  }
  return [r,g,b];
}

function clusterPixels (pixels) {
  var shape = pixels.shape.slice();
  var totalPixels = shape[0] * shape[1]
  var sampleRate = 10
  var samplePixels = totalPixels / sampleRate

  var pxArray = []
  for(var x = 0; x < shape[0]; x+=sampleRate) {
    for(var y = 0; y < shape[1]; y+=sampleRate) {
      pxArray.push([
        pixels.get(x,y,0),
        pixels.get(x,y,1),
        pixels.get(x,y,2)
      ])
    }
  }

  var clusters = kmeans(pxArray, 5)
  clusters = clusters.map(function(cluster){
    var r = 0;
    var g = 0;
    var b = 0;
    cluster.forEach(function(color){
      r += color[0]
      g += color[1]
      b += color[2]
    })
    return [r/cluster.length, g/cluster.length, b/cluster.length, cluster.length].map(Math.round)  
  })
  clusters = clusters.sort(function(a, b){
    return b[3] - a[3]
  })

  return clusters
}