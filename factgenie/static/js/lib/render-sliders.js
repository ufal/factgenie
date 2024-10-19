$(document).ready(function () {
    function adjustSliders() {
        // find all the sliders of class ".slider-crowdsourcing"
        const sliders = $('.slider-crowdsourcing');

        for (let i = 0; i < sliders.length; i++) {
            const slider = sliders.eq(i);
            // we need to find the datalist of the slider
            // the slider has an id "slider-crowdsourcing-{i}"
            // the datalist has an id "slider-crowdsourcing-{i}-values"
            // we cannot use the iteration index since there may be also selectboxes
            const sliderValues = $(`#${slider.attr('id')}-values`);
            const sliderOptions = sliderValues.find('option');

            if (sliderOptions.length > 1) {
                const firstOption = sliderOptions.first();
                const lastOption = sliderOptions.last();

                const firstOptionCenter = firstOption.position().left + firstOption.outerWidth() / 2;
                const lastOptionCenter = lastOption.position().left + lastOption.outerWidth() / 2;

                const sliderWidth = lastOptionCenter - firstOptionCenter;

                // hardcoded padding of the outside box;
                const leftOffset = firstOptionCenter - 22;

                slider.css('width', `${sliderWidth}px`);
                slider.css('left', `${leftOffset}px`);
            }
        }
    }

    // Initial adjustment
    adjustSliders();

    // Adjust on window resize
    $(window).resize(adjustSliders);
});