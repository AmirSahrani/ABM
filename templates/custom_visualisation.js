function handleHover(canvas, portrayal) {
    const tooltip = document.getElementById("tooltip");
    const rect = canvas.getBoundingClientRect();

    canvas.addEventListener("mousemove", function(evt) {
        const mouseX = evt.clientX - rect.left;
        const mouseY = evt.clientY - rect.top;

        let found = false;

        portrayal.forEach(function(obj) {
            var x = obj.x * canvas.width;
            var y = obj.y * canvas.height;
            var r = obj.r || 0.5;

            if (Math.abs(mouseX - x) < r * canvas.width && Math.abs(mouseY - y) < r * canvas.height) {
                found = true;
                tooltip.style.display = 'block';
                tooltip.style.left = (evt.clientX + 10) + 'px';
                tooltip.style.top = (evt.clientY + 10) + 'px';
                tooltip.innerHTML = `
                    ${obj.hover_text ? obj.hover_text : ''}
                    ${obj.spice_level ? '<br>Spice Level: ' + obj.spice_level : ''}
                `;
            }
        });

        if (!found) {
            tooltip.style.display = 'none';
        }
    });
}




var ContinuousVisualization = function(height, width, context) {
    var canvas = document.createElement("canvas");
    canvas.height = height;
    canvas.width = width;
    context.appendChild(canvas);

    var context = canvas.getContext("2d");

    this.draw = function(portrayal) {
        context.clearRect(0, 0, width, height);

        portrayal.forEach(function(obj) {
            var x = obj.x * width;
            var y = obj.y * height;
            var r = obj.r || 0.5;
            var color = obj.Color;
            var shape = obj.Shape;
            var filled = obj.Filled;

            if (shape === "circle") {
                context.beginPath();
                context.arc(x, y, r * width, 0, 2 * Math.PI);
                if (filled) {
                    context.fillStyle = color;
                    context.fill();
                } else {
                    context.strokeStyle = color;
                    context.stroke();
                }
            } else if (shape === "rect") {
                var w = obj.w || 1;
                var h = obj.h || 1;
                if (filled) {
                    context.fillStyle = color;
                    context.fillRect(x - w / 2, y - h / 2, w * width, h * height);
                } else {
                    context.strokeStyle = color;
                    context.strokeRect(x - w / 2, y - h / 2, w * width, h * height);
                }
            }
        });

        // Call handleHover only once
        handleHover(canvas, portrayal);
    };
};
