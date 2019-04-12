let model;
let signaturePad;

$(function() {

	createModel();

	$("#clear").click(function() {

		signaturePad.clear();
		model.clear();
		clearResult();

	});

	$("#resetTrigger").click(function() {

		signaturePad.clear();
		clearResult();

	});

	signaturePad = new SignaturePad( document.getElementById( 'signature-pad' ), {

		minWidth: 10,
		backgroundColor: 'rgba(255, 255, 255, 0)',
		penColor: 'rgb(214, 253, 255)',
		onEnd: executePredict

	} );

});

function createModel() {

	let container = document.getElementById( "modelArea" );

	model = new TSP.models.Sequential( container, {

		animeTime: 200,
		stats: true

	} );

	model.add( new TSP.layers.GreyscaleInput() );

	model.add( new TSP.layers.Padding2d() );

	model.add( new TSP.layers.Conv2d({
		initStatus: "open"
	}) );

	model.add( new TSP.layers.Pooling2d() );

	model.add( new TSP.layers.Conv2d() );

	model.add( new TSP.layers.Pooling2d() );

	model.add( new TSP.layers.Dense() );

	model.add( new TSP.layers.Dense() );

	model.add( new TSP.layers.Output1d( {
		
		outputs: [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" ],
		initStatus: "open",

	} ) );

	model.load( {

		type: "tfjs",
		url: '../../assets/model/lenet/mnist.json',

	} );

	model.init( function() {

		$.ajax({
			url: "../../assets/data/digit/5.json",
			type: 'GET',
			async: true,
			dataType: 'json',
			success: function (data) {

				model.predict( data );

			}
		});

	} );

}

function executePredict() {

	let canvas = document.getElementById( "signature-pad" );
	let context = canvas.getContext( '2d' );
	let imgData = context.getImageData( 0, 0, canvas.width, canvas.height );

	let signatureData = [];

	for ( let i = 0; i < 224; i += 8 ) {

		for ( let j = 3; j < 896; j += 32 ) {

			signatureData.push( imgData.data[ 896 * i + j ] / 255 );

		}

	}

	model.predict( signatureData, function( predictResult ) {

		let index = getMaxConfidentNumber(predictResult);
		clearResult();
		highLightResult(index);

	});

}

function getMaxConfidentNumber( predictResult ) {

	let maxIndex = 0;

	for ( let i = 1; i < predictResult.length; i ++ ) {

		maxIndex = predictResult[ i ] > predictResult[ maxIndex ] ? i : maxIndex;

	}

	return maxIndex;

}

function highLightResult( index ) {

	let idString = "result" + index;
	$("#" + idString).css( {
		"color": "#D6FDFF"
	} );

}

function clearResult() {

	for ( let i = 0; i < 10; i ++ ) {

		let idString = "result" + i;
		$("#" + idString).css({
			"color": "#456989"
		});

	}

}