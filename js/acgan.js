let model;
let labelIndex = 0;

$(function() {

	createModel();

	$("#labelHolder > button").click(function() {
		clearPreviousIndex();
		labelIndex = parseInt($(this).attr("data-index"));

		$(this).css( {
			"background-color": "#D6FDFF",
			"color": "#000000"
		} );

	});

	$("#generateButton").click(function() {
		generateDigitImage();
	});

});

function createModel() {

	let container = document.getElementById( "modelArea" );

	model = new TSP.models.Sequential(container, {

		stats: true,
		feedInputs: [ 0 ]

	});

	// input_1: [1,100]
	model.add(new TSP.layers.Input1d({
		shape: [100]
	}));

	// 0
	// output: (128*7*7=) 6272
	model.add(new TSP.layers.Dense({
		units: 6272,
		paging: true,
		segmentLength: 400,
		overview: true
	}));

	// 1
	// output: 128*7*7 = 6272
	model.add(new TSP.layers.Reshape({
		targetShape: [7, 7, 128]
	}));

	// 2
	// output: 128*14*14 = 25088
	model.add(new TSP.layers.UpSampling2d({
		size: [2, 2]
	}));

	// 3
	// output: 128*14*14 = 25088
	model.add(new TSP.layers.Conv2d({
		kernelSize: 3,
		filters: 128,
		strides: 1,
		padding: "same"
	}));

	// 4
	// output: 128*28*28 = 100352
	model.add(new TSP.layers.UpSampling2d({
		size: [2, 2]
	}));

	// 5
	// output: 64*28*28 = 50176
	model.add(new TSP.layers.Conv2d({
		kernelSize: 3,
		filters: 64,
		strides: 1,
		padding: "same"
	}));

	// 6
	// output: 1*28*28 = 784
	model.add(new TSP.layers.Conv2d({
		kernelSize: 3,
		filters: 1,
		strides: 1,
		padding: "same",
		name: "digitLayer"
	}));

	model.load({
		type: "tfjs",
		url: '../../assets/model/acgan/model.json',
		multiInputs: true,
		inputShapes: [[100], [1]]
	});

	model.init(function() {

		generateDigitImage();
		$("#loadingPad").hide();

	});

}

function clearPreviousIndex() {

	$("#labelHolder > button").each(function() {
		$(this).css({
			"background-color": "#233D45",
			"color": "#D6FDFF"
		});
	})

}

function generateDigitImage() {

	let randomData = tf.randomNormal([1,100]).dataSync();
	model.predict( [randomData, [labelIndex]] );
	let digitLayer = model.getLayerByName("digitLayer");
	renderDigitCanvas(digitLayer.neuralValue);
}

function renderDigitCanvas(digitDataArray) {
	// let digitCanvas = $("#generatedDigit");

    let inputData = digitDataArray;
    // let inputData =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0.011764705882352941,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00784313725490196,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0.5411764705882353,0,0.9607843137254902,0.32941176470588235,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3764705882352941,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0.0392156862745098,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0392156862745098,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0.8274509803921568,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0.03137254901960784,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6392156862745098,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

    let c = document.getElementById("generatedDigit");
    let ctx = c.getContext("2d");
    let imgData = ctx.createImageData(224, 224);

    let count = 0;
    let tempArray = [];
    let rearrangedArray = [];

    for (let j = 0; j < inputData.length; j++) {
        if (count === 28) {
            rearrangedArray.push(tempArray);
            tempArray = [];
            count = 0;
        }
        tempArray.push(inputData[j]);
        count++;
    }

    function resultArrayWithZeros(dimensions) {
        let array = [];
        for (let i = 0; i < dimensions[0]; ++i) {
            array.push(dimensions.length === 1 ? 0 : resultArrayWithZeros(dimensions.slice(1)));
        }
        return array;
    }

    let row = 224;
    let col = 224;
    let resultArrayAllZeros = resultArrayWithZeros([row, col]);

    for (let m = 0; m < rearrangedArray.length; m++) {
        for (let n = 0; n < rearrangedArray[0].length; n++) {
            resultArrayMinMax(m, n, rearrangedArray[m][n]);
        }
    }

    function resultArrayMinMax(a, b, num) {
        let rowMultiple = 8, colMultiple = 8;
        let rowMin = a * rowMultiple;
        let rowMax = a * rowMultiple + rowMultiple - 1;
        let colMin = b * colMultiple;
        let colMax = b * colMultiple + colMultiple - 1;

        for (let p = rowMin; p <= rowMax; p++) {
            for (let q = colMin; q <= colMax; q++) {
                resultArrayAllZeros[p][q] = num;
            }
        }
    }

    let resultArray = [];

    for (let h = 0; h < resultArrayAllZeros.length; h++) {
        resultArray = resultArray.concat(resultArrayAllZeros[h]);
    }

    for (let u = 0, w = 0; w < resultArray.length; u += 4, w++) {
        imgData.data[u + 0] = 0;
        imgData.data[u + 1] = 0;
        imgData.data[u + 2] = 0;
        imgData.data[u + 3] = resultArray[w] * 255;
    }

    ctx.putImageData(imgData, 0, 0);

}