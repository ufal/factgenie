// Initialize an array to store annotations
let annotations = [];

// Function to update the outputarea with the annotations JSON
function updateOutputArea() {
    // include only the fields reason, text and type in the stringified JSON and omit the rest
    const output = { "annotations": annotations.map(a => ({ "reason": a.reason, "text": a.text, "type": a.type })) };
    $('#outputarea').text(JSON.stringify(output, null, 4));
}

// Function to handle adding a new annotation
function onAnnotationAdded(annotation) {
    // Add the annotation to the annotations array
    annotations.push(annotation);
    const categoryName = getAnnotationSpanCategories()[annotation.type].name;

    // Add a new row to the errorarea
    const row = $(`
         <tr id="error-row-${annotation.id}">
             <td>${categoryName}</td>
             <td>${annotation.text}</td>
             <td>
                 <input type="text" class="form-control reason-input" data-id="${annotation.id}" placeholder="Enter reason">
             </td>
         </tr>
     `);
    $('#errorarea').append(row);

    // Attach event listener to the reason input
    row.find('.reason-input').on('input', function () {
        const id = $(this).data('id');
        const reason = $(this).val();
        // Update the reason in the annotations array
        const annotation = annotations.find(a => a.id === id);
        if (annotation) {
            annotation.reason = reason;
            updateOutputArea();
        }
    });

    // Update the output area
    updateOutputArea();
}

// Function to handle deleting an annotation
function onAnnotationDeleted(annotation) {
    const annotationId = annotation.cid;

    // Remove the annotation from the annotations array
    annotations = annotations.filter(a => a.id !== annotationId);

    // Remove the corresponding row from errorarea
    $(`#error-row-${annotationId}`).remove();

    // Update the output area
    updateOutputArea();
}

function pasteExampleIntoPrompt() {
    const exampleData = $('#example-data').val();
    const exampleText = $('#example-text').val();
    const output = $('#outputarea').text();
    const prompt = `\n\n*Example:*\ninput:\n\`\`\`\n${exampleData}\n\`\`\`\ntext:\n\`\`\`\n${exampleText}\n\`\`\`\noutput:\n\`\`\`\n${output}\n`;
    $('#prompt-template').val($('#prompt-template').val() + prompt);
    $('#exampleAnnotation').modal('hide');
}

function createButtons() {
    // if buttons already exist, remove them
    $('#buttons-area').empty();

    const annotationSpanCategories = getAnnotationSpanCategories();

    for (const [idx, category] of Object.entries(annotationSpanCategories)) {
        const input = $('<input>', {
            type: "radio",
            class: "btn-check btn-outline-secondary btn-err-cat",
            name: "btnradio",
            id: `btnradio${idx}`,
            autocomplete: "off",
            "data-cat-idx": idx
        });

        const label = $('<label>', {
            class: "btn btn-err-cat-label me-1",
            for: `btnradio${idx}`,
            style: `background-color: ${category.color};`
        }).text(category.name);

        if (idx == 0) {
            input.attr('checked', 'checked');
            label.addClass('active');
        }

        $('#buttons-area').append(input);
        $('#buttons-area').append(label);
    }

    // Add eraser button
    const eraserInput = $('<input>', {
        type: "radio",
        class: "btn-check btn-outline-secondary btn-eraser",
        name: "btnradio",
        id: "btnradioeraser",
        autocomplete: "off",
        "data-cat-idx": "-1"
    });

    const eraserLabel = $('<label>', {
        class: "btn btn-err-cat-label ms-auto",
        for: "btnradioeraser",
        style: "background-color: #FFF; color: #000 !important;"
    }).text("Erase mode");

    $('#buttons-area').append(eraserInput);
    $('#buttons-area').append(eraserLabel);

    // Event handlers
    $(".btn-err-cat, .btn-eraser").change(function () {
        if (this.checked) {
            const cat_idx = $(this).attr("data-cat-idx");
            spanAnnotator.setCurrentAnnotationType(cat_idx);
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
}

function initAnnotation() {
    $("#nextsteparea").show();
    $('#errorarea').empty();
    $('#outputarea').empty();
    $('#annotationarea').empty();

    const annotationSpanCategories = getAnnotationSpanCategories();

    spanAnnotator.init(
        "words",
        true,
        annotationSpanCategories
    );

    const exampleText = $('#example-text').val();
    const p = $('<p>', { class: 'annotatable-paragraph' }).html(exampleText);

    // add paragraph to #annotationarea
    $('#annotationarea').append(p);

    spanAnnotator.addDocument("p-example", p, true);
    spanAnnotator.setCurrentAnnotationType(0);

    spanAnnotator.addEventListener('annotationAdded', function (data) {
        onAnnotationAdded(data.annotation);
    });

    spanAnnotator.addEventListener('annotationRemoved', function (data) {
        data.removedAnnotations.forEach(annotation => {
            onAnnotationDeleted(annotation);
        });
    });
}

function checkAndOpenModal() {
    const annotationSpanCategories = getAnnotationSpanCategories();
    if (annotationSpanCategories.length == 0) {
        alert("Please add at least one annotation span category.");
        return;
    }
    createButtons();
    var modal = new bootstrap.Modal('#exampleAnnotation');
    modal.show();
}