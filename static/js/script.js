$(document).ready(function () {
    let slider = $("#PanNum");
    let input = $("#RangeSliderValue");

    slider.on("input", function () {
        input[0].value = $(this)[0].value;
        var value = ($(this)[0].value - $(this)[0].min) / ($(this)[0].max - $(this)[0]
            .min) * 100;
        $(this)[0].style.background = 'linear-gradient(to right, #82CFD0 0%, #82CFD0 ' +
            value +
            '%, #d7dcdf ' +
            value +
            '%, #d7dcdf 100%)'
    });


    input.on("input", function () {
        slider[0].value = $(this)[0].value;
        var value = ($(this)[0].value - $(this)[0].min) / ($(this)[0].max - $(this)[0]
            .min) * 100;
        slider[0].style.background = 'linear-gradient(to right, #82CFD0 0%, #82CFD0 ' +
            value +
            '%, #d7dcdf ' +
            value +
            '%, #d7dcdf 100%)'
    });
    input.on("change", function () {
        if (parseInt($(this)[0].value) > $(this)[0].max) {
            $(this)[0].value = $(this)[0].max
        }
        if (parseInt($(this)[0].value) < $(this)[0].min) {
            $(this)[0].value = $(this)[0].min
        }
    })
})

$(document).ready(function () {
    $("input[name$='RadioChoose']").click(function () {
        var ButtonValue = $(this).val();
        if (ButtonValue === "File") {
            $("#FileForm").hide();
            $("#PasteForm").show();
        } else {
            $("#PasteForm").hide();
            $("#FileForm").show();
        }
    });
});