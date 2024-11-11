var current_example_idx = 0;
var selected_campaigns = [];
var collapsed_boxes = [];
var splitInstance = Split(['#centerpanel', '#rightpanel'], {
    sizes: [66, 33],
    gutterSize: 1,
});

function changeDataset() {
    $("#dataset-spinner").show();
    const dataset = $('#dataset-select').val();

    // set available splits in #split-select
    $('#split-select').empty();
    for (const split of datasets[dataset].splits) {
        $('#split-select').append(`<option value="${split}">${split}</option>`);
    }
    const split = $('#split-select').val();

    current_example_idx = 0;
    fetchExample(dataset, split, current_example_idx);
    $("#page-input").val(current_example_idx);
}

function changeSplit() {
    $("#dataset-spinner").show();
    const dataset = $('#dataset-select').val();
    const split = $('#split-select').val();
    current_example_idx = 0;
    fetchExample(dataset, split, current_example_idx);
    $("#page-input").val(current_example_idx);
}

function changeExample(dataset, split, example_idx) {
    $("#dataset-spinner").show();
    $('#dataset-select').val(dataset);
    $('#split-select').empty();
    for (const split of datasets[dataset].splits) {
        $('#split-select').append(`<option value="${split}">${split}</option>`);
    }
    $('#split-select').val(split);
    current_example_idx = example_idx;
    fetchExample(dataset, split, example_idx);
    $("#page-input").val(example_idx);
}


function createOutputBox(content, exampleLevelFields, annId, setup_id) {
    var card = $('<div>', { class: `card output-box generated-output-box box-${setup_id} box-${annId} box-${setup_id}-${annId}` });

    var annotationBadge = (annId !== "original") ? `<span class="small"><i class="fa fa-pencil"></i> ${annId}</span>` : ""
    var headerHTML = `<div class="d-flex justify-content-between">
    <span class="small">${setup_id}</span>
    ${annotationBadge}
    </div>
    `
    var cardHeader = $('<div>', { class: "card-header card-header-collapse small", "data-bs-toggle": "collapse", "data-bs-target": `#out-${setup_id}-${annId}` }).html(headerHTML);
    var cardBody = $('<div>', { class: "card-body show", id: `out-${setup_id}-${annId}`, "aria-expanded": "true" });
    // var cardTitle = $('<h5>', { class: "card-title" }).text(setup_id);
    var cardText = $('<div>', { class: "card-text mt-2" }).html(content);

    // cardBody.append(cardTitle);
    cardBody.append(cardText);
    card.append(cardHeader);
    card.append(cardBody);

    if (exampleLevelFields !== null) {
        cardBody.append(exampleLevelFields);
    }

    return card;
}

function generateAnnotatorShortId(campaign_id, annotator_group) {
    const ann_id = `${campaign_id}-ann${annotator_group}`;
    return ann_id;
}

function createOutputBoxes(generated_outputs, highlight_setup_id) {
    // clear the output area
    $("#outputarea").empty();

    // sort outputs by setup id
    generated_outputs.sort(function (a, b) {
        return a.setup_id.localeCompare(b.setup_id);
    });

    // find all campaign ids in output annotations
    const annIds = new Map();
    generated_outputs.forEach(output => {
        output.annotations.forEach(annotation => {
            const campaign_id = annotation.campaign_id;
            const annotator_group = annotation.annotator_group;
            const ann_id = generateAnnotatorShortId(campaign_id, annotator_group);

            annIds.set(ann_id, { "campaign_id": campaign_id, "annotator_group": annotator_group });
        });
    });
    const selectBox = $("#annotations-select");
    // clear the selectbox
    selectBox.empty();

    // add an option for each campaign id
    for (const [ann_id, { campaign_id, annotator_group }] of annIds) {
        const button = $(`<button type="button" class="btn btn-sm btn-primary btn-ann-select" data-ann="${ann_id}">${ann_id}</button>`);
        button.on('click', function () {
            $(this).toggleClass('active');
            updateDisplayedAnnotations();
        });
        selectBox.append(button);
    }
    if (annIds.size > 0) {
        $("#setuparea").show();
    } else {
        $("#setuparea").hide();
    }

    // add the annotated outputs
    for (const output of generated_outputs) {
        var groupDiv = $('<div>', { class: `output-group box-${output.setup_id} d-inline-flex gap-2` });
        groupDiv.appendTo("#outputarea");

        const plain_output = getAnnotatedOutput(output, "original", null);
        card = createOutputBox(plain_output, null, "original", output.setup_id);
        card.appendTo(groupDiv);

        for (const [annId, { campaign_id, annotator_group }] of annIds) {
            annotations = output.annotations.filter(a => a.campaign_id == campaign_id && a.annotator_group == annotator_group)[0];

            const annotated_output = getAnnotatedOutput(output, annId, annotations);
            const exampleLevelFields = getExampleLevelFields(annotations);

            card = createOutputBox(annotated_output, exampleLevelFields, annId, output.setup_id);
            card.appendTo(groupDiv);
            card.hide();
        }

        // Highlight the output box if highlight_setup_id matches
        if (highlight_setup_id && highlight_setup_id === output.setup_id) {
            card.addClass('border border-primary border-2');
            groupDiv.css({ animation: 'jump-out 0.5s ease' });
        }
    }
}

function fetchExample(dataset, split, example_idx) {
    // change the URL so that it shows the permalink
    const newUrl = `${url_prefix}/browse?dataset=${dataset}&split=${split}&example_idx=${example_idx}`;

    // the check prevents being stuck at the same URL
    if (!window.location.href.includes(newUrl)) {
        history.pushState(null, '', newUrl);
    }
    $.get(`${url_prefix}/example`, {
        "dataset": dataset,
        "example_idx": example_idx,
        "split": split,
    }, function (data) {
        if (data.error !== undefined) {
            console.log(data.error);
            alert(data.error);
            return;
        }
        $("#dataset-spinner").hide();
        $("#examplearea").html(data.html);

        showRawData(data);

        total_examples = datasets[dataset].example_count[split];
        $("#total-examples").html(total_examples - 1);

        createOutputBoxes(data.generated_outputs, window.highlight_setup_id);
        showSelectedCampaigns();
        updateDisplayedAnnotations();

        window.highlight_setup_id = null;
    }).fail(function (response) {
        console.log(response);
        alert("Failed to fetch example.");
    });
}

function getAnnotatedOutput(output, annId, annotations) {
    const setup_id = output.setup_id;

    // replace newlines with any spaces around them with <br>
    const content = output.output.replace(/\\n/g, '<br>');

    var placeholder = $('<pre>', { id: `out-${setup_id}-${annId}-placeholder`, class: `font-mono out-placeholder out-${annId}-placeholder` });
    var annotated_content;

    if (annotations !== null && annotations !== undefined) {
        const annotation_span_categories = annotations.annotation_span_categories;
        annotated_content = highlightContent(content, annotations, annotation_span_categories);
    } else {
        // we do not have outputs for the particular campaign -> grey out the text
        if (annId != "original") {
            placeholder.css("color", "#c2c2c2");
        }
        annotated_content = content;
    }
    placeholder.html(annotated_content);
    // placeholder.hide();
    return placeholder;
}

function getExampleLevelFields(annotations) {
    if (annotations === null || annotations === undefined) {
        return null;
    }
    // show `outputs.flags`, `outputs.options`, and `outputs.textFields`
    var flags = annotations.flags;
    var options = annotations.options;
    var textFields = annotations.textFields;

    var html = $('<div>', { class: "p-2 extra-fields" });

    if (flags !== undefined && flags.length > 0) {
        var flagsDiv = $('<div>', { class: "small" });
        // flagsDiv.append($('<span class="badge bg-secondary">').html("Flags"));
        for (const flag of flags) {
            var labelDiv = $('<div>', { class: "small text-muted " }).text(`${flag.label}`);
            var valueDiv = $('<div>', { class: "small mb-1 fw-bold" }).text(`${flag.value}`);

            flagsDiv.append(labelDiv);
            flagsDiv.append(valueDiv);
        }
        html.append(flagsDiv);
    }

    if (options !== undefined && options.length > 0) {
        var optionsDiv = $('<div>', { class: "small" });

        for (const option of options) {
            var labelDiv = $('<div>', { class: "small text-muted" }).text(`${option.label}`);
            var valueDiv = $('<div>', { class: "small mb-1 fw-bold" }).text(`${option.value}`);

            optionsDiv.append(labelDiv);
            optionsDiv.append(valueDiv);
        }
        html.append(optionsDiv);
    }

    if (textFields !== undefined && textFields.length > 0) {
        var textFieldsDiv = $('<div>', { class: "small" });

        for (const textField of textFields) {
            var labelDiv = $('<div>', { class: "small text-muted" }).text(`${textField.label}`);
            var valueDiv = $('<div>', { class: "small mb-1 fw-bold" }).text(`${textField.value}`);

            textFieldsDiv.append(labelDiv);
            textFieldsDiv.append(valueDiv);
        }
        html.append(textFieldsDiv);
    }
    return html;
}

function goToPage(page) {
    current_example_idx = Math.min(page, total_examples - 1);
    current_example_idx = Math.max(0, current_example_idx);

    const dataset = $('#dataset-select').val();
    const split = $('#split-select').val();

    fetchExample(dataset, split, current_example_idx);

    $("#page-input").val(current_example_idx);
}

// Our custom function to highlight the text spans using the collected annotations
// Here we do *not* use the YPet library, but directly work with HTML
function highlightContent(content, annotations, annotation_span_categories) {
    let offset = 0; // Track cumulative offset
    const annotationSet = annotations.annotations;

    // sort by start
    annotationSet.sort(function (a, b) {
        return a.start - b.start;
    });
    var html = content;

    annotationSet.forEach(annotation => {
        const annotationType = annotation.type;

        if (!(annotationType in annotation_span_categories)) {
            console.log("Warning: annotation type not found in annotation_span_categories: " + annotationType);
            return;
        }
        const color = annotation_span_categories[annotationType].color;
        const text = annotation.text.trimEnd();

        const start = annotation.start + offset;
        const end = start + text.length;

        const error_name = annotation_span_categories[annotationType].name;
        const note = annotation.reason || annotation.note;
        let tooltip_text;

        if (note !== undefined && note !== "" && note !== null) {
            tooltip_text = `${error_name} (${note})`;
        } else {
            tooltip_text = `${error_name}`;
        }

        const spanId = `span-${start}-${end}`;
        const spanContent = `<span id="${spanId}" style="margin-right: 0px;background-color: ${color};" data-bs-toggle="tooltip" data-bs-placement="top" title="${tooltip_text}">${text}</span>`;

        html = html.slice(0, start) + spanContent + html.slice(end);
        // Update the offset
        offset += spanContent.length - text.length;
    });
    return html;
}

function showRawData(data) {
    var rawDataStr = JSON.stringify(data.raw_data, null, 2).replace(/\\n/g, '<br>');

    if (rawDataStr[0] == '"') {
        // remove the first and last double quotes
        rawDataStr = rawDataStr.slice(1, -1);
    }
    rawDataStr = rawDataStr.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    $("#rawarea").html(`<pre>${rawDataStr}</pre>`);
}


function showSelectedCampaigns() {
    // if the annotator is still among the values, restore it
    // prevent resetting to the first annotation when switching examples
    $(".btn-ann-select").each(function () {
        if (selected_campaigns.includes($(this).data('ann'))) {
            $(this).addClass("active").trigger("change");
        } else {
            $(this).removeClass("active");
        }
    });


}

function toggleRaw() {
    // toggle display: none on rawarea and examplearea
    $("#rawarea").toggle();
    $("#examplearea").toggle();
}


function updateDisplayedAnnotations() {
    const activeButtons = $('.btn-ann-select.active');
    selected_campaigns = activeButtons.map(function () {
        return $(this).data('ann');
    }).get();
    // hide all placeholders
    $(".output-box").hide();

    // if no campaign ids, show the original output
    if (selected_campaigns.length == 0) {
        $(".box-original").show();
    }
    for (const campaign_id of selected_campaigns) {
        // show the selected annotator
        $(`.box-${campaign_id}`).show();
    }

    restoreCollapsedStates();
    enableTooltips();
}

$('#page-input').keypress(function (event) {
    // Enter = Go to page
    if (event.keyCode == 13) {
        goToBtn();
    }
});

$("#dataset-select").on("change", changeDataset);
$("#split-select").on("change", changeSplit);

$(document).keydown(function (event) {
    const key = event.key;

    if (key === "ArrowRight") {
        event.preventDefault();
        nextBtn();
    } else if (key === "ArrowLeft") {
        event.preventDefault();
        prevBtn();
    }
});

// checking whether the user is navigating to an example using history: if so, we need to load the particular example
window.addEventListener('popstate', function (event) {
    if (window.location.pathname === `/browse`) {
        const params = new URLSearchParams(window.location.search);
        const dataset = params.get('dataset');
        const split = params.get('split');
        const example_idx = params.get('example_idx');
        if (dataset && split && example_idx) {
            changeExample(dataset, split, example_idx);
        }
    }
});


$(document).ready(function () {
    if (window.display_example != null) {
        const e = window.display_example;
        changeExample(e.dataset, e.split, e.example_idx);
    }
    else {
        // select the first dataset from the selectbox
        $("#dataset-select").val(
            $("#dataset-select option:first").val()
        ).trigger("change");
        $("#page-input").val(current_example_idx);
    }

    enableTooltips();
});

// Restore collapsed states after page change/content load
function restoreCollapsedStates() {
    collapsed_boxes.forEach(targetId => {
        const element = document.getElementById(targetId);
        if (element) {
            const bsCollapse = new bootstrap.Collapse(element, {
                toggle: false
            });
            bsCollapse.hide();
        }
    });
}

// Store collapsed state when boxes are toggled
document.addEventListener('shown.bs.collapse', function (e) {
    const targetId = e.target.id;
    collapsed_boxes = collapsed_boxes.filter(id => id !== targetId);
});

document.addEventListener('hidden.bs.collapse', function (e) {
    const targetId = e.target.id;
    if (!collapsed_boxes.includes(targetId)) {
        collapsed_boxes.push(targetId);
    }
});