$(document).ready(function () {
    const sliders = $('.slider-crowdsourcing');

    sliders.each(function (index) {
        const sliderId = $(this).attr('id');
        const valueDisplayId = `${sliderId}-value`;

        // Set initial value
        $(`#${valueDisplayId}`).text($(`#${valueDisplayId}`).data('default-value'));

        // Update value on slider change
        $(this).on('input change', function () {
            $(`#${valueDisplayId}`).text($(this).val());
        });
    });
});