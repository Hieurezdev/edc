$(document).ready(function () {
    $(document).ready(function () {
        // Hide all fieldsets except the first one
        $("fieldset").hide();
        $("fieldset:first").show();
        updateProgressBar(0); // Initial progress bar setup

        // Next button click
        $(".next").click(function () {
            var currentFieldset = $(this).parent();
            var nextFieldset = currentFieldset.next("fieldset");

            // Check if a model is selected
            if (currentFieldset.is(':first-child') && !$('#model').val()) {
                alert("Please select a model.");
                return;
            }

            // Fade out current fieldset and fade in next one
            currentFieldset.fadeOut(400, function () {
                nextFieldset.fadeIn(400);
                updateProgressBar(nextFieldset.index());
            });
        });

        // Previous button click
        $(".previous").click(function () {
            var currentFieldset = $(this).parent();
            var previousFieldset = currentFieldset.prev("fieldset");

            // Fade out current fieldset and fade in previous one
            currentFieldset.fadeOut(400, function () {
                previousFieldset.fadeIn(400);
                updateProgressBar(previousFieldset.index());
            });
        });

        // Update the progress bar
        function updateProgressBar(index) {
            $("#progressbar li").removeClass("active");
            $("#progressbar li").each(function (i) {
                if (i < index) {
                    $(this).addClass("active");
                }
            });
        }
    });

    // Toggle switch between pages
    $('#switch').on('click', function () {
        $(this).toggleClass('active');
        $('body').toggleClass('toggle-on-bg');

        if ($(this).hasClass('active')) {
            $('#normalizationPage').hide();
            $('#dictionaryPage').show();
        } else {
            $('#normalizationPage').show();
            $('#dictionaryPage').hide();
        }
    });

    // Handle Load Model button click
    $('#loadModel').click(function () {
        const percent = $('#percent').val();
        const model = $('#model').val();
        $.post('/load_model', { percent: percent, model: model }, function (response) {
            $('#modelInfo').html('<div class="' + (response.status === 'success' ? 'info' : 'error') + '">' + response.message + '</div><br><pre>' + response.log + '</pre>').addClass('visible');
        });
    });

    // Hide result section initially
    $('.result').hide();

    // Handle form submission and normalization process
    $('#normalizeText').click(function (event) {
        event.preventDefault();
        var inputText = $('#inputText').val();
        var model = $('#model').val();
        var percent = $('#percent').val();

        // Ajax request to backend normalization API
        $.ajax({
            url: '/normalize_text',
            type: 'POST',
            data: { input_text: inputText },
            success: function (response) {
                if (response.status === 'success') {
                    // Show the result section
                    $('.result').show();
                    // Display the normalized text in the output textarea
                    $('#outputText').val(response.normalized_text);

                    // Update the highlighted text in the div
                    $('#highlightedText').html(response.highlighted_text);

                    // Update the detection info in the div
                    $('#detection_info tbody').html(response.detection_info);
                } else {
                    // Handle error case
                    alert(response.message);
                }
            },
            error: function () {
                alert('Error during the request. Please try again.');
            }
        });
    });

    // Glow effect for the toggle button
    $('.toggle').click(function (e) {
        e.preventDefault(); // The flicker is a codepen thing
        $(this).toggleClass('toggle-on');
    });
    
    
    // Search functionality on button click
    $('#searchBtn').click(function () {
        const input = $('.inputValue').val().toLowerCase(); // Get the search input

        // Search the dictionary first
        $.post('/lookup_dict', { nsw: input }, function(response) {
                if (response.status === 'success') {
                    // Display the result from the dictionary
                    $('.section3').hide();
                    $('.value').text(response.word);
                    $('.definition').text(response.definition);
                    $('.abbreviation').text(response.abbreviation);
                    $('.example').text(response.example);
                    $('.section2').show(); // Show the results
                } 
            }).fail(function() {
                // If not found in dictionary, call the backend to get ChatGPT response
                    $.post('/lookup_API', { nsw: input }, function (response) {
                        if (response.status === 'success') {
                            const chatgptResponse = response.message; // Get response message
                            // Display the response
                            $('.section2').hide();
                            $('.response').html(chatgptResponse); // Display the response
                            $('.section3').show(); // Show the results
                        } else {
                            alert('No results found and ChatGPT response failed: ' + response.message);
                        }
                    });
            });
    });
});


    
