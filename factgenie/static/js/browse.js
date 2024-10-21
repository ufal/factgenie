const sizes = [66, 33];

const generated_outputs = window.generated_outputs;

var current_example_idx = 0;
var selected_campaigns = [];
var splitInstance = Split(['#centerpanel', '#rightpanel'], {
    sizes: sizes,
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
    // used only for direct links
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


function createOutputBox(content, exampleLevelFields, campaign_id, setup_id) {
    var card = $('<div>', { class: `card output-box generated-output-box box-${setup_id} box-${campaign_id} box-${setup_id}-${campaign_id}` });

    var annotationBadge = (campaign_id !== "original") ? `<span class="small"><i class="fa fa-pencil"></i> ${campaign_id}</span>` : ""
    var headerHTML = `<div class="d-flex justify-content-between">
    <span class="small">${setup_id}</span>
    ${annotationBadge}
    </div>
    `

    var cardHeader = $('<div>', { class: "card-header card-header-collapse small", "data-bs-toggle": "collapse", "data-bs-target": `#out-${setup_id}-${campaign_id}` }).html(headerHTML);
    var cardBody = $('<div>', { class: "card-body show", id: `out-${setup_id}-${campaign_id}`, "aria-expanded": "true" });
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

function createOutputBoxes(generated_outputs) {
    // clear the output area
    $("#outputarea").empty();

    // sort outputs by setup id
    generated_outputs.sort(function (a, b) {
        return a.setup_id.localeCompare(b.setup_id);
    });

    // find all campaign ids in output annotations
    const campaign_ids = new Set();

    generated_outputs.forEach(output => {
        output.annotations.forEach(annotation => {
            campaign_ids.add(annotation.metadata.id);
        });
    });
    const selectBox = $("#annotations-select");
    // clear the selectbox
    selectBox.empty();

    // add an option for each campaign id
    for (const campaign_id of campaign_ids) {
        const button = $(`<button type="button" class="btn btn-sm btn-primary btn-ann-select" data-ann="${campaign_id}">${campaign_id}</button>`);
        button.on('click', function () {
            $(this).toggleClass('active');
            updateDisplayedAnnotations();
        });
        selectBox.append(button);
    }

    if (campaign_ids.size > 0) {
        $("#setuparea").show();
    } else {
        $("#setuparea").hide();
    }

    // add the annotated outputs
    for (const output of generated_outputs) {
        var groupDiv = $('<div>', { class: `output-group box-${output.setup_id} d-inline-flex gap-2` });
        groupDiv.appendTo("#outputarea");

        const plain_output = getAnnotatedOutput(output, "original");
        card = createOutputBox(plain_output, null, "original", output.setup_id);
        card.appendTo(groupDiv);

        for (const campaign_id of campaign_ids) {
            const annotated_output = getAnnotatedOutput(output, campaign_id);
            const exampleLevelFields = getExampleLevelFields(output, campaign_id);
            card = createOutputBox(annotated_output, exampleLevelFields, campaign_id, output.setup_id);
            card.appendTo(groupDiv);
            card.hide();
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
            alert(data.error);
            return;
        }
        $("#dataset-spinner").hide();
        $("#examplearea").html(data.html);

        showRawData(data);

        total_examples = data.total_examples;
        $("#total-examples").html(total_examples - 1);

        createOutputBoxes(data.generated_outputs);
        showSelectedCampaigns();
        updateDisplayedAnnotations();
    });
}

function getAnnotatedOutput(output, campaign_id) {
    const setup_id = output.setup_id;
    // replace newlines with any spaces around them with <br>
    const content = output.out.replace(/\\n/g, '<br>');

    // if the campaign_id is in output.annotations, show the annotated content
    const annotations_campaign = output.annotations.filter(a => a.metadata.id == campaign_id);

    var placeholder = $('<pre>', { id: `out-${setup_id}-${campaign_id}-placeholder`, class: `font-mono out-placeholder out-${campaign_id}-placeholder` });
    var annotated_content;

    if (annotations_campaign.length > 0) {
        const annotations = annotations_campaign[0];
        const annotation_span_categories = annotations.metadata.config.annotation_span_categories;

        annotated_content = highlightContent(content, annotations, annotation_span_categories);
    } else {
        // we do not have outputs for the particular campaign -> grey out the text
        if (campaign_id != "original") {
            placeholder.css("color", "#c2c2c2");
        }
        annotated_content = content;
    }
    placeholder.html(annotated_content);
    // placeholder.hide();
    return placeholder;
}

function getExampleLevelFields(output, campaign_id) {
    const fieldsCampaign = output.annotations.filter(a => a.metadata.id == campaign_id)[0];

    if (fieldsCampaign === undefined) {
        return null;
    }
    // show `outputs.flags`, `outputs.options`, and `outputs.textFields`
    var flags = fieldsCampaign.flags;
    var options = fieldsCampaign.options;
    var textFields = fieldsCampaign.textFields;

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
