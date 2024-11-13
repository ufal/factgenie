const sizes = [50, 50];
const annotator_id = window.annotator_id;
const metadata = window.metadata;

var current_example_idx = 0;
var annotation_set = window.annotation_set;

const total_examples = annotation_set.length;

var examples_cached = {};

var splitInstance = Split(['#centerpanel', '#rightpanel'], {
    sizes: sizes,
    gutterSize: 1,
    onDrag: function () {
        // trigger a resize update on slider inputs when the handle is dragged
        // not a perfect solution, but it works
        $('.slider-crowdsourcing').each(function () {
            $(this).css('width', "80%");
        });
    }
});

function clearExampleLevelFields() {
    // uncheck all checkboxes
    $(".crowdsourcing-flag input[type='checkbox']").prop("checked", false);

    // reset all options to the first value
    $(".crowdsourcing-option select").val(0);
    $(".crowdsourcing-option input[type='range']").val(0);

    // clear the values in text inputs
    $(".crowdsourcing-text input[type='text']").val("");
}


function collectFlags() {
    const flags = [];
    $(".crowdsourcing-flag").each(function () {
        const label = $(this).find("label").text().trim();
        const value = $(this).find("input[type='checkbox']").prop("checked");
        flags.push({
            label: label,
            value: value
        });
    });
    return flags;
}

function collectOptions() {
    const options = [];

    $(".crowdsourcing-option").each(function (x) {
        if ($(this).hasClass("option-select")) {
            const type = "select";
            const label = $(this).find("label").text().trim();
            const index = $(this).find("select").val();
            const value = $(this).find("select option:selected").text();

            const optionList = $(this).find("select option").map(function () {
                return $(this).text();
            }).get();
            options.push({ type: type, label: label, index: index, value: value, optionList: optionList });
        } else if ($(this).hasClass("option-slider")) {
            const type = "slider";
            const label = $(this).find("label").text();
            const index = $(this).find("input[type='range']").val();
            const value = $(this).find("datalist option")[index].value;
            const optionList = $(this).find("datalist option").map(function () {
                return $(this).val();
            }).get();
            options.push({ type: type, label: label, index: index, value: value, optionList: optionList });
        }
    });
    return options;
}

function collectTextFields() {
    const textFields = [];

    $(".crowdsourcing-text").each(function (x) {
        const label = $(this).find("label").text().trim();
        const value = $(this).find("input[type='text']").val();
        textFields.push({ label: label, value: value });
    });
    return textFields;
}


function fetchAnnotation(dataset, split, setup_id, example_idx, annotation_idx) {
    return new Promise((resolve, reject) => {
        $.get(`${url_prefix}/example`, {
            "dataset": dataset,
            "example_idx": example_idx,
            "split": split,
            "setup_id": setup_id
        }, function (data) {
            $('<div>', {
                id: `out-text-${annotation_idx}`,
                class: `annotate-box `,
                style: 'display: none;'
            }).appendTo('#outputarea');

            // we have always only a single generated output here
            data.generated_outputs = data.generated_outputs[0];
            examples_cached[annotation_idx] = data;
            resolve();
        }).fail(function () {
            reject();
        });
    });
}


function goToAnnotation(example_idx) {
    $(".page-link").removeClass("bg-active");
    $(`#page-link-${example_idx}`).addClass("bg-active");

    $(".annotate-box").hide();
    $(`#out-text-${example_idx}`).show();

    const data = examples_cached[example_idx];
    $("#examplearea").html(data.html);

    const flags = annotation_set[example_idx].flags;
    const options = annotation_set[example_idx].options;
    const textFields = annotation_set[example_idx].textFields;

    clearExampleLevelFields();

    if (flags !== undefined) {
        $(".crowdsourcing-flag").each(function (i) {
            $(this).find("input[type='checkbox']").prop("checked", flags[i]["value"]);
        });
    }

    if (options !== undefined) {
        for (const [i, option] of Object.entries(options)) {
            const div = $(`.crowdsourcing-option:eq(${i})`);
            // we can have either a select or a slider (we can tell by `type`)
            // we need to set the option defined by `index`
            if (option.type == "select") {
                div.find("select").val(option.index);
            }
            if (option.type == "slider") {
                div.find("input[type='range']").val(option.index);
            }
        }
    }

    if (textFields !== undefined) {
        for (const [i, textField] of Object.entries(textFields)) {
            $(`.crowdsourcing-text input:eq(${i})`).val(textField.value);
        }
    }

}

function goToPage(page) {
    const example_idx = current_example_idx;

    current_example_idx = page;
    current_example_idx = mod(current_example_idx, total_examples);

    saveCurrentAnnotations(example_idx);
    goToAnnotation(current_example_idx);
}

function loadAnnotations() {
    $("#dataset-spinner").show();

    const promises = [];
    const annotation_span_categories = metadata.config.annotation_span_categories;

    // prefetch the examples for annotation: we need them for YPet initialization
    for (const [annotation_idx, example] of Object.entries(annotation_set)) {
        const dataset = example.dataset;
        const split = example.split;
        const example_idx = example.example_idx;
        const setup_id = example.setup_id;

        const promise = fetchAnnotation(dataset, split, setup_id, example_idx, annotation_idx);
        promises.push(promise);
    }
    Promise.all(promises)
        .then(() => {

            YPet.addInitializer(function (options) {
                /* Configure the # and colors of Annotation types (minimum 1 required) */
                YPet.AnnotationTypes = new AnnotationTypeList(annotation_span_categories);
                var regions = {};
                var paragraphs = {};

                for (const [annotation_idx, data] of Object.entries(examples_cached)) {

                    var p = new Paragraph({ 'text': data.generated_outputs.output, 'granularity': metadata.config.annotation_granularity });

                    paragraphs[`p${annotation_idx}`] = p;
                    regions[`p${annotation_idx}`] = `#out-text-${annotation_idx}`;

                    const li = $('<li>', { class: "page-item" });
                    const a = $('<a>', { class: "page-link bg-incomplete", style: "min-height: 28px;", id: `page-link-${annotation_idx}` }).text(annotation_idx);
                    li.append(a);
                    $("#nav-example-cnt").append(li);

                    // switch to the corresponding example when clicking on the page number
                    $(`#page-link-${annotation_idx}`).click(function () {
                        goToPage(annotation_idx);
                    });
                }
                YPet.addRegions(regions);

                for (const [p, p_obj] of Object.entries(paragraphs)) {
                    YPet[p].show(new WordCollectionView({ collection: p_obj.get('words') }));

                    YPet[p].currentView.collection.parentDocument.get('annotations').on('remove', function (model, collection) {
                        if (collection.length == 0) {
                            collection = [];
                        }
                    });
                }
                goToAnnotation(0);
            });
            YPet.start();

            $("#hideOverlayBtn").attr("disabled", false);
            $("#hideOverlayBtn").html("View the annotation page");
        })
        .catch((e) => {
            // Handle errors if any request fails
            console.error("One or more requests failed.");
            // Log the error
            console.error(e);

        })
        .finally(() => {
            // This block will be executed regardless of success or failure
            $("#dataset-spinner").hide();
        });
}

function markAnnotationAsComplete() {
    $('#page-link-' + current_example_idx).removeClass("bg-incomplete");
    $('#page-link-' + current_example_idx).addClass("bg-complete");

    // if all the examples are annotated, post the annotations
    if ($(".bg-incomplete").length == 0) {
        saveCurrentAnnotations(current_example_idx);

        // show the `submit` button
        $("#submit-annotations-btn").show();

        // scroll to the top
        $('html, body').animate({
            scrollTop: $("#submit-annotations-btn").offset().top
        }, 500);

    } else if (current_example_idx < total_examples - 1) {
        // annotations will be saved automatically
        nextBtn();
    }
}

function saveCurrentAnnotations(example_idx) {
    var collection = YPet[`p${example_idx}`].currentView.collection.parentDocument.get('annotations').toJSON();
    annotation_set[example_idx]["annotations"] = collection;
    annotation_set[example_idx]["flags"] = collectFlags();
    annotation_set[example_idx]["options"] = collectOptions();
    annotation_set[example_idx]["textFields"] = collectTextFields();
}

function submitAnnotations(campaign_id) {
    // remove `words` from the annotations: they are only used by the YPet library
    for (const example of annotation_set) {
        for (const annotation of example.annotations) {
            delete annotation.words;
        }
    }

    $.post({
        url: `${url_prefix}/submit_annotations`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaign_id: metadata.id,
            annotator_id: annotator_id,
            annotation_set: annotation_set
        }
        ),
        success: function (response) {
            console.log(response);
            window.onbeforeunload = null;

            if (response.success !== true) {
                $("#error-message").html(response.error);
                $("#overlay-fail").show();
            } else {
                $("#final-message").html(response.message);
                $("#overlay-end").show();
            }
        },
        error: function (response) {
            console.log(response);
            $("#overlay-fail").show();
        }
    });
}


$("#hideOverlayBtn").click(function () {
    $("#overlay-start").fadeOut();
});

$(".btn-err-cat").change(function () {
    if (this.checked) {
        const cat_idx = $(this).attr("data-cat-idx");
        YPet.setCurrentAnnotationType(cat_idx);
    }
});

$('.btn-check').on('change', function () {
    $('.btn-check').each(function () {
        const label = $(`label[for=${this.id}]`);
        if (this.checked) {
            label.addClass('active');
        } else {
            label.removeClass('active');
        }
    });
});

$(document).ready(function () {
    loadAnnotations();
    $("#total-examples").html(total_examples - 1);
    enableTooltips();
});

window.onbeforeunload = function () {
    return "Are you sure you want to reload the page? Your work will be lost.";
}