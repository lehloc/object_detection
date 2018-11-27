$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        // var form_data = new FormData($('#upload-file')[0]);
        var file = $("#imageUpload")[0].files[0];
		if (!file) {
			return;
	    }
        // Show loading animation
        $(this).hide();
        $('.loader').show();
		new ImageCompressor(file, {
			quality: .7,
			width:  502,
			height: 894,
			success(result) {
			    const formData = new FormData();
			    formData.append('file', result, result.name);
                console.log(result);
				console.log(result.name);
        // Make prediction by calling api /predict
				$.ajax({
					type: 'POST',
					url: '/predict',
					data: formData,
					contentType: false,
					cache: false,
					processData: false,
					async: true,
					success: function (data) {
						// Get and display the result
						$('.loader').hide();
						$('#result').fadeIn(600);
						$('#result').text(' Result:  ' + data);
						console.log('Success!');
					},
				});
			},
			error(e) {
			    console.log(e.message);
			},
		});
    });
});
