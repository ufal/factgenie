class SpanAnnotator {
    constructor() {
        this.documents = new Map();
        this.currentType = 0;
        this.annotationTypes = [];
        this.isSelecting = false;
        this.startSpan = null;
        this.throttledFindClosestSpan = this._throttle(
            (objectId, x, y) => this._findClosestSpan(objectId, x, y),
            100
        );
        this.eraserPreviewActive = false;
        this.rightClickPreviewActive = false;
        this.eventListeners = new Map();
        this.history = new Map(); // Map of document ID -> array of history states
        this.currentHistoryIndex = new Map(); // Map of document ID -> current history index
    }

    init(granularity, overlapAllowed, annotationTypes) {
        this.granularity = granularity;
        this.overlapAllowed = overlapAllowed;
        this.annotationTypes = annotationTypes;
        return this;
    }

    setCurrentAnnotationType(type) {
        // make sure that type is integer
        this.currentType = parseInt(type);

        // Get all annotatable paragraph boxes
        const paragraphBoxes = $('.annotate-box');
        const paragraphs = $('.annotatable-paragraph');

        if (type === -2) {
            // Select mode - use text cursor
            paragraphs.addClass('select-mode-enabled');
            paragraphBoxes.css('cursor', 'text');
        } else if (type === -1) {
            paragraphs.removeClass('select-mode-enabled');
            // Eraser mode - use eraser cursor
            paragraphBoxes.css('cursor', 'pointer');
        } else if (type != null) {
            paragraphs.removeClass('select-mode-enabled');
            // Annotation mode - use colored brush cursor based on category
            const color = this.annotationTypes[type]?.color;
            if (color) {
                // Create colored cursor style
                paragraphBoxes.css({
                    'cursor': `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='25' height='25' viewBox='0 0 328.862 328.862'%3E%3Cg%3E%3Cpath fill='${encodeURIComponent(color)}' d='M251.217,195.25L56.286,69.063c-4.609-2.984-10.48-3.21-15.308-0.591c-4.826,2.62-7.835,7.667-7.844,13.158l-0.375,232.206c-0.01,6.371,4.006,12.054,10.016,14.172c1.633,0.576,3.315,0.854,4.981,0.854c4.464,0,8.802-1.997,11.704-5.617l71.455-89.101l113.645-11.378c6.34-0.635,11.587-5.206,13.085-11.398C259.143,205.176,256.566,198.712,251.217,195.25z'/%3E%3C/g%3E%3C/svg%3E") 7 7, pointer`
                });
            }
        }
    }

    addEventListener(eventName, callback) {
        if (!this.eventListeners.has(eventName)) {
            this.eventListeners.set(eventName, []);
        }
        this.eventListeners.get(eventName).push(callback);
    }

    _addToHistory(objectId) {
        const doc = this.documents.get(objectId);
        if (!this.history.has(objectId)) {
            this.history.set(objectId, []);
            this.currentHistoryIndex.set(objectId, -1);
        }

        // Remove any future history after current index
        const currentIndex = this.currentHistoryIndex.get(objectId);
        this.history.get(objectId).splice(currentIndex + 1);

        // Add new state
        this.history.get(objectId).push({
            annotations: JSON.parse(JSON.stringify(doc.annotations)) // Deep copy
        });
        this.currentHistoryIndex.set(objectId, this.currentHistoryIndex.get(objectId) + 1);
    }

    undo(objectId) {
        if (!this.history.has(objectId)) return;

        const currentIndex = this.currentHistoryIndex.get(objectId);
        if (currentIndex < 0) return;

        // Restore previous state
        const previousState = this.history.get(objectId)[currentIndex];
        const doc = this.documents.get(objectId);
        doc.annotations = JSON.parse(JSON.stringify(previousState.annotations));

        // Update index
        this.currentHistoryIndex.set(objectId, currentIndex - 1);

        // Rerender
        this._renderAnnotations(objectId);
        this.emit('annotationUndone', { objectId });
    }

    emit(eventName, data) {
        if (this.eventListeners.has(eventName)) {
            this.eventListeners.get(eventName).forEach(callback => callback(data));
        }
    }

    addDocument(objectId, htmlObject, dynamic = false) {
        const $element = $(htmlObject);
        const text = $element.text();
        const spans = this._createSpans(text);

        $element.html(spans);

        this.documents.set(objectId, {
            element: $element,
            text: text,
            annotations: []
        });

        if (dynamic) {
            this._attachEventHandlers(objectId);
        }
    }

    // used to add annotations externally
    addAnnotations(objectId, annotations) {
        const doc = this.documents.get(objectId);
        doc.annotations = annotations;
        this._renderAnnotations(objectId);
    }

    getAnnotations(objectId) {
        return this.documents.get(objectId)?.annotations || [];
    }

    _createSpans(text) {
        let currentIndex = 0;
        if (this.granularity === 'words') {
            const parts = text.split(/(\s+)/);

            return parts.map((part, arrayIndex) => {
                if (arrayIndex % 2 === 1) {
                    return ''; // Remove standalone whitespace spans
                }

                const whitespace = arrayIndex < parts.length - 1 ? parts[arrayIndex + 1] : '';
                const fullContent = String(part) + whitespace;
                const span = `<span class="annotatable" 
                    data-index="${currentIndex}" 
                    data-content="${part}"
                    data-whitespace="${whitespace}">${part}<span class="whitespace">${whitespace === '\n' ? '<br>' : whitespace}</span></span>`;

                currentIndex += fullContent.length;
                return span;
            }).join('');
        } else {
            return text.split('').map(char => {
                const span = `<span class="annotatable" 
                    data-index="${currentIndex}"
                    data-content="${char}"
                    >${char === '\n' ? '<br>' : char}</span>`;
                currentIndex += 1;
                return span;
            }).join('');
        }
    }

    _attachEventHandlers(objectId) {
        const doc = this.documents.get(objectId);
        const $elementPar = doc.element;

        // Change element to parent's div
        const $element = $elementPar.parent();

        $element.on('selectstart', (e) => {
            // Only prevent selection when not in select mode
            if (this.currentType !== -2) {
                e.preventDefault();
            }
        });

        $element.on('contextmenu', (e) => {
            e.preventDefault();
        });

        $element.on('mousedown', (e) => {
            // Skip annotation logic in select mode
            if (this.currentType === -2) {
                return;
            }

            if (e.button === 2) { // Right click
                const span = this._findClosestSpan(objectId, e.clientX, e.clientY);
                if (span) {
                    this._removeAnnotation(objectId, span);
                }
                return;
            }
            this.isSelecting = true;
            this.startSpan = this._findClosestSpan(objectId, e.clientX, e.clientY);
        });

        $element.on('mousemove', (e) => {
            if (e.buttons === 2) { // Right button pressed
                const closestSpan = this._findClosestSpan(objectId, e.clientX, e.clientY);
                if (closestSpan) {
                    this._previewEraserEffect(objectId, closestSpan);
                    this.rightClickPreviewActive = true;
                }
                return;
            }
            if (this.isSelecting) {
                const closestSpan = this.throttledFindClosestSpan(objectId, e.clientX, e.clientY);
                if (closestSpan) {
                    this._updateHighlight(objectId, this.startSpan, closestSpan);
                }
            } else if (this.currentType === -1) {
                const closestSpan = this._findClosestSpan(objectId, e.clientX, e.clientY);
                if (closestSpan) {
                    this._previewEraserEffect(objectId, closestSpan);
                }
            }
        });


        $element.on('mouseup', (e) => {
            if (!this.isSelecting) return;
            this.isSelecting = false;
            const endSpan = this._findClosestSpan(objectId, e.clientX, e.clientY);

            if (this.startSpan && endSpan) {
                if (this.currentType === -1) {
                    this._removeAnnotation(objectId, endSpan);
                } else {
                    this._createAnnotation(objectId, this.startSpan, endSpan);
                }
            }
        });

        // Handle mouse leaving the element
        $element.on('mouseleave', () => {
            if (this.isSelecting) {
                this.isSelecting = false;
                this._renderAnnotations(objectId);
            }
            if (this.eraserPreviewActive) {
                this._clearEraserPreview(objectId);
            }
            if (this.rightClickPreviewActive) {
                this._clearEraserPreview(objectId);
                this.rightClickPreviewActive = false;
            }
        });
    }

    _previewEraserEffect(objectId, $span) {
        const doc = this.documents.get(objectId);
        const position = parseInt($span.data('index'));

        this.eraserPreviewActive = true;

        // Clear previous preview
        this._clearEraserPreview(objectId);

        // Find annotations that would be removed
        const affectedAnnotations = doc.annotations.filter(a =>
            position >= a.start && position < a.start + a.text.length);

        // Highlight spans for each affected annotation
        affectedAnnotations.forEach(ann => {
            $('.annotatable', doc.element).each((_, span) => {
                const $span = $(span);
                const idx = parseInt($span.data('index'));
                if (idx >= ann.start && idx < ann.start + ann.text.length) {

                    // make the tokens more transparent by applying a filter
                    $span.css('filter', 'opacity(0.5)');
                }
            });
        });
    }

    _clearEraserPreview(objectId) {
        if (!this.eraserPreviewActive) return;

        const doc = this.documents.get(objectId);
        // remove the filter from all spans
        $('.annotatable', doc.element).css('filter', '');
        this.eraserPreviewActive = false;
    }

    _throttle(func, limit) {
        let lastResult;
        let lastRun;

        return function (...args) {
            if (!lastRun || Date.now() - lastRun >= limit) {
                lastResult = func.apply(this, args);
                lastRun = Date.now();
            }
            return lastResult;
        }
    }

    _hasExistingAnnotations(doc, startIdx, endIdx) {
        return doc.annotations.some(ann => {
            // Check if any part of the new annotation overlaps with existing ones
            const annotationEnd = ann.start + ann.text.length - 1;
            return (startIdx <= annotationEnd && endIdx >= ann.start);
        });
    }

    _findClosestSpan(objectId, x, y) {
        const doc = this.documents.get(objectId);
        let closestSpan = null;
        let minDistance = Infinity;

        // if we already are over a span, return that
        const currentSpan = $(document.elementFromPoint(x, y)).closest('.annotatable');
        if (currentSpan.length > 0) {
            return currentSpan;
        }


        $('.annotatable', doc.element).each((_, span) => {
            const $span = $(span);
            const rect = span.getBoundingClientRect();
            const left = rect.left;
            const top = rect.top;
            const right = left + $span.width();
            const bottom = top + $span.height();

            const dx = Math.abs((left + right) / 2 - x);
            const dy = Math.abs((top + bottom) / 2 - y);
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < minDistance) {
                minDistance = distance;
                closestSpan = $span;
            }
        });

        return closestSpan;
    }

    _updateHighlight(objectId, $start, $end) {
        const doc = this.documents.get(objectId);
        $('.annotatable', doc.element).css('background', '');

        const startIdx = parseInt($start.data('index'));
        const endIdx = parseInt($end.data('index'));
        const [min, max] = [Math.min(startIdx, endIdx), Math.max(startIdx, endIdx)];

        $('.annotatable', doc.element).each((_, span) => {
            const $span = $(span);
            const idx = parseInt($span.data('index'));
            if (idx >= min && idx <= max) {
                const color = this.annotationTypes[this.currentType]?.color;
                $span.css({
                    'background': `linear-gradient(0deg, ${color} 2px, transparent 2px)`,
                    'background-position': '0 100%',
                    'padding-bottom': '3px'
                });
            }
        });
    }

    _createAnnotation(objectId, $start, $end) {
        this._addToHistory(objectId);

        const doc = this.documents.get(objectId);
        const startIdx = parseInt($start.data('index'));
        const endIdx = parseInt($end.data('index')); // Remove the content length adjustment
        const [min, max] = [Math.min(startIdx, endIdx), Math.max(startIdx, endIdx)];

        // Get the actual end position by adding length of the last token
        const maxWithLength = max + String($end.data('content')).length - 1;


        // Check for exactly matching annotations
        const isExisting = doc.annotations.some(ann =>
            ann.start === min &&
            ann.start + ann.text.length === maxWithLength + 1 &&
            ann.type === this.currentType
        );

        // Check for overlap if not allowed
        if ((!this.overlapAllowed && this._hasExistingAnnotations(doc, min, maxWithLength)) || isExisting) {
            this._renderAnnotations(objectId);
            return;
        }

        const text = doc.text.substring(min, maxWithLength + 1);
        const id = Math.random().toString(36).substring(2, 10);

        const annotation = {
            type: this.currentType,
            text: text,
            start: min,
            id: id
        };

        doc.annotations.push(annotation);
        this._renderAnnotations(objectId);
        this.emit('annotationAdded', { objectId, annotation });
    }

    _removeAnnotation(objectId, $span) {
        this._addToHistory(objectId);

        const doc = this.documents.get(objectId);
        const position = parseInt($span.data('index'));

        const removedAnnotations = doc.annotations.filter(a =>
            position >= a.start && position < a.start + a.text.length);

        doc.annotations = doc.annotations.filter(a =>
            position < a.start || position >= a.start + a.text.length);

        this._renderAnnotations(objectId);
        this.emit('annotationRemoved', { objectId, removedAnnotations });
    }

    _renderAnnotations(objectId) {
        const doc = this.documents.get(objectId);
        $('.annotatable', doc.element).each((_, span) => {
            const $span = $(span);
            const position = parseInt($span.data('index'));

            const spanAnnotations = doc.annotations.filter(a =>
                position >= a.start && position < a.start + a.text.length);

            // Reset styling
            $span.attr('style', '');
            $('.whitespace', $span).removeClass('whitespace-hidden');

            if (spanAnnotations.length > 0) {
                const content = String($span.data('content'));
                const isLastInAnyAnnotation = spanAnnotations.some(ann =>
                    position + content.length >= ann.start + ann.text.length);
                const hasMultipleAnnotations = spanAnnotations.length > 1;

                if (isLastInAnyAnnotation && !hasMultipleAnnotations) {
                    // add right padding to the last span in the annotation
                    if (this.granularity === 'words') {
                        const char = $span.attr('data-whitespace');
                        if (char !== '\n') {
                            $('.whitespace', $span).addClass('whitespace-hidden');
                            $span.css('margin-right', '9px');
                        }
                    }
                }

                const gradients = spanAnnotations.map((ann, i) => {
                    const offset = i * 3;
                    const color = this.annotationTypes[ann.type].color;
                    return `linear-gradient(0deg, ${color} ${4 + offset}px, transparent ${4 + offset}px)`;
                });

                $span.css({
                    'background': gradients.join(', '),
                    'background-position': '0 100%',
                    'line-height': '8px',
                    'padding-bottom': `${3 + (spanAnnotations.length - 1) * 3}px`,
                    'color': this.annotationTypes[spanAnnotations[spanAnnotations.length - 1].type].color,
                    'font-weight': 'bold'
                });

                // const note = annotation.reason || annotation.note;
                // const tooltip_text = note ? `${error_name} (${note})` : error_name;

                const tooltipText = spanAnnotations.map(ann => {
                    const name = this.annotationTypes[ann.type].name;
                    const note = ann.reason || ann.note;
                    return note ? `${name} (${note})` : name;
                }).join(', ');

                // data-bs-toggle="tooltip" data-bs-placement="top" title="${tooltip_text}"
                $span.attr('data-bs-toggle', 'tooltip');
                $span.attr('data-bs-placement', 'top');
                $span.attr('title', tooltipText);

            }
        });
    }
}

const spanAnnotator = new SpanAnnotator();