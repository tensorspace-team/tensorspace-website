let model, conv2d;

$(function() {

	$("#more").click(function() {

		if ($("#nav-collapse").is(":visible")) {
			$("#nav-collapse").slideUp(function() {
				$("#smallGuide").show();
			});
		} else {
			$("#nav-collapse").slideDown();
			$("#smallGuide").hide();
		}

	});

	$("#downloadNav").click(function () {

		$("#downloadNav").addClass("now");

		$('html, body').animate({
			scrollTop: $("#download").offset().top
		}, 2000);
	});

    $('#toPlayground').mouseover(function() {
		$(this).find("img").attr("src", "assets/img/index/playgroundIcon_darkB.png").delay(400);
	}).mouseout(function() {
		$(this).find("img").attr("src", "assets/img/index/playgroundIcon.png");
	});

    $("#galleryLenet").hover(function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_LeNet.gif");
	}, function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_LeNet.jpg");
	});

	$("#galleryAlexnet").hover(function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_AlexNet.gif");
	}, function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_AlexNet.jpg");
	});

	$("#galleryYolo").hover(function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_yolov2.gif");
	}, function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_yolov2.jpg");
	});

	$("#galleryAcgan").hover(function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_acgan.gif");
	}, function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_acgan.jpg");
	});

	$("#galleryResnet").hover(function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_ResNet.gif");
	}, function(){
		$(this).find(".img33").attr("src", "./assets/img/index/gallery_ResNet.jpg");
	});

	$("#galleryMore").hover(function(){
		$(this).find(".img33").attr("src", "./assets/img/index/more.gif");
	}, function(){
		$(this).find(".img33").attr("src", "./assets/img/index/more.jpg");
	});

	createModel();

});

function createModel() {

	let windowWidth = document.body.clientWidth;

	let container;

	if ( windowWidth > 490 ) {
		container = document.getElementById( "modelArea" );
	} else {
		container = document.getElementById( "smallModelArea" );
	}

	model = new TSP.models.Sequential( container, {

		animeTime: 1600

	} );

	model.add( new TSP.layers.GreyscaleInput() );

	model.add( new TSP.layers.Padding2d() );

	conv2d =  new TSP.layers.Conv2d();

	model.add( conv2d );

	model.add( new TSP.layers.Pooling2d() );

	model.add( new TSP.layers.Conv2d() );

	model.add( new TSP.layers.Pooling2d() );

	model.add( new TSP.layers.Dense() );

	model.add( new TSP.layers.Dense() );

	model.add( new TSP.layers.Output1d( {
		
		outputs: [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" ],
		initStatus: "open"

	} ) );

	model.load( {

		type: "tfjs",
		url: './assets/model/lenet/mnist.json',

	} );

	model.init(function() {

		$.ajax({
			url: "./assets/data/digit/5.json",
			type: 'GET',
			async: true,
			dataType: 'json',
			success: function (data) {

				model.predict( data );
				conv2d.openLayer();
				launchPredictor();

			}
		});

	});

}

function launchPredictor() {

	setInterval(function() {

		let digit = Math.floor(10 * Math.random());
		$.ajax({
			url: "./assets/data/digit/" + digit + ".json",
			type: 'GET',
			async: true,
			dataType: 'json',
			success: function (data) {

				model.predict( data );

			}
		});

	}, 3000);

}