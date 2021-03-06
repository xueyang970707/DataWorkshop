/*******************************************************************************
 * Copyright (c) 2013 Evan Carey,
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *    Evan Carey
 *    Bryan Grohman
 *******************************************************************************/

//
// Treemap utilities
//

var TreemapUtils = TreemapUtils || {};
TreemapUtils.KeySpline = function (e) {
    function t(e, t) {
        return 1 - 3 * t + 3 * e
    }

    function n(e, t) {
        return 3 * t - 6 * e
    }

    function o(e) {
        return 3 * e
    }

    function i(e, i, r) {
        return ((t(i, r) * e + n(i, r)) * e + o(i)) * e
    }

    function r(e, i, r) {
        return 3 * t(i, r) * e * e + 2 * n(i, r) * e + o(i)
    }

    function s(e) {
        for (var t = e, n = 0; 4 > n; ++n) {
            var o = r(t, a, d);
            if (0 == o) return t;
            var s = i(t, a, d) - e;
            t -= s / o
        }
        return t
    }

    var a = e.mX1 || 0, l = e.mY1 || 0, d = e.mX2 || 1, h = e.mY2 || 1;
    this.get = function (e) {
        return a == l && d == h ? e : i(s(e), l, h)
    }
}, TreemapUtils.Easing = {
    ease: {mX1: .25, mY1: .1, mX2: .25, mY2: 1},
    linear: {mX1: 0, mY1: 0, mX2: 1, mY2: 1},
    "ease-in": {mX1: .42, mY1: 0, mX2: 1, mY2: 1},
    "ease-out": {mX1: 0, mY1: 0, mX2: .58, mY2: 1},
    "ease-in-out": {mX1: .42, mY1: 0, mX2: .58, mY2: 1}
}, TreemapUtils.sumArray = function () {
    function e(e, t) {
        return e + t
    }

    return function (t) {
        return t.reduce(e)
    }
}(), TreemapUtils.deepCopy = function (e) {
    if ("[object Array]" === Object.prototype.toString.call(e)) {
        for (var t = [], n = 0, o = e.length; o > n; n++) t[n] = arguments.callee(e[n]);
        return t
    }
    if ("object" == typeof e) {
        var n, t = {};
        for (n in e) t[n] = arguments.callee(e[n]);
        return t
    }
    return e
}, TreemapUtils.changeColor = function (e, t, n) {
    e = e.replace(/^\s*|\s*$/, ""), e = e.replace(/^#?([a-f0-9])([a-f0-9])([a-f0-9])$/i, "#$1$1$2$2$3$3");
    var o = Math.round(256 * t) * (n ? -1 : 1),
        i = e.match(new RegExp("^rgba?\\(\\s*(\\d|[1-9]\\d|1\\d{2}|2[0-4][0-9]|25[0-5])\\s*,\\s*(\\d|[1-9]\\d|1\\d{2}|2[0-4][0-9]|25[0-5])\\s*,\\s*(\\d|[1-9]\\d|1\\d{2}|2[0-4][0-9]|25[0-5])(?:\\s*,\\s*(0|1|0?\\.\\d+))?\\s*\\)$", "i")),
        r = i && null !== i[4] ? i[4] : null,
        s = i ? [i[1], i[2], i[3]] : e.replace(/^#?([a-f0-9][a-f0-9])([a-f0-9][a-f0-9])([a-f0-9][a-f0-9])/i, function (e, t, n, o) {
            return parseInt(t, 16) + "," + parseInt(n, 16) + "," + parseInt(o, 16)
        }).split(/,/);
    return i ? "rgb" + (null !== r ? "a" : "") + "(" + Math[n ? "max" : "min"](parseInt(s[0], 10) + o, n ? 0 : 255) + ", " + Math[n ? "max" : "min"](parseInt(s[1], 10) + o, n ? 0 : 255) + ", " + Math[n ? "max" : "min"](parseInt(s[2], 10) + o, n ? 0 : 255) + (null !== r ? ", " + r : "") + ")" : ["#", Math[n ? "max" : "min"](parseInt(s[0], 10) + o, n ? 0 : 255).toString(16).lpad("0", 2), Math[n ? "max" : "min"](parseInt(s[1], 10) + o, n ? 0 : 255).toString(16).lpad("0", 2), Math[n ? "max" : "min"](parseInt(s[2], 10) + o, n ? 0 : 255).toString(16).lpad("0", 2)].join("")
}, TreemapUtils.lighterColor = function (e, t) {
    return TreemapUtils.changeColor(e, t, !1)
}, TreemapUtils.darkerColor = function (e, t) {
    return TreemapUtils.changeColor(e, t, !0)
}, TreemapUtils.rgb2hex = function (e) {
    var t = "#" + (e[2] | e[1] << 8 | e[0] << 16).toString(16).lpad("0", 6);
    return t
}, TreemapUtils.avgRgb = function (e) {
    return Math.floor(TreemapUtils.sumArray(e) / 3)
}, TreemapUtils.hex2rgb = function (e) {
    return e.replace(/^#?([a-f0-9][a-f0-9])([a-f0-9][a-f0-9])([a-f0-9][a-f0-9])/i, function (e, t, n, o) {
        return parseInt(t, 16) + "," + parseInt(n, 16) + "," + parseInt(o, 16)
    }).split(/,/)
}, TreemapUtils.squarify = function (e, t) {
    var n, o = function (e) {
        this.setX = function (t) {
            e[2] -= t - e[0], e[0] = t
        }, this.setY = function (t) {
            e[3] -= t - e[1], e[1] = t
        }, this.getX = function () {
            return e[0]
        }, this.getY = function () {
            return e[1]
        }, this.getW = function () {
            return e[2]
        }, this.getH = function () {
            return e[3]
        }, this.getWidth = function () {
            return Math.min(e[2], e[3])
        }
    }, i = function (e, t) {
        var n = Math.max.apply(null, e), o = Math.min.apply(null, e), i = TreemapUtils.sumArray(e), r = i * i,
            s = t * t;
        return Math.max(s * n / r, r / (s * o))
    }, r = function (e) {
        var t, n, o, i = h.getX(), r = h.getY(), s = i + h.getW(), a = r + h.getH();
        if (h.getW() < h.getH()) {
            for (t = Math.ceil(TreemapUtils.sumArray(e) / h.getW()), r + t >= a && (t = a - r), n = 0; n < e.length; n++) o = Math.ceil(e[n] / t), (i + o > s || n + 1 === e.length) && (o = s - i), l.push([i, r, o, t]), i += o;
            h.setY(r + t)
        } else {
            for (t = Math.ceil(TreemapUtils.sumArray(e) / h.getH()), i + t >= s && (t = s - i), n = 0; n < e.length; n++) o = Math.ceil(e[n] / t), (r + o > a || n + 1 === e.length) && (o = a - r), l.push([i, r, t, o]), r += o;
            h.setX(i + t)
        }
    }, s = function (e) {
        var t = [];
        if (t.push(e.shift()), 0 === e.length) return t;
        var n = t.slice(), o = h.getWidth();
        do {
            if (n.push(e[0]), !(i(t, o) > i(n, o))) break;
            t = n.slice(), e.shift()
        } while (e.length > 0);
        return t
    }, a = function (e) {
        do r(s(e)); while (e.length > 0)
    }, l = [], d = [];
    if (e[2] <= 0 || e[3] <= 0) for (n = 0; n < t.length; n++) l.push(e.slice()); else {
        d = t.map(function (t) {
            return t * e[2] * e[3]
        });
        var h = new o(e.slice());
        a(d)
    }
    return l
}, function (e) {
    e.widget("ui.treemap", {
        options: {
            dimensions: [600, 400],
            colorStops: [{val: 0, color: "#08f"}, {val: .5, color: "#03f"}, {val: 1, color: "#005"}],
            colorResolution: 1024,
            naColor: "#000",
            innerNodeHeaderHeightPx: 12,
            innerNodeHeaderLabeller: function (e, t, n, o) {
                e.rect(t[0], t[1], t[2], t[3]), e.clip(), e.fillStyle = "#555", e.font = "0.625em Verdana, Geneva, sans-serif", e.fillText(o, t[0], t[1] + 10)
            },
            innerNodeHeaderGradient: function (e, t) {
                var n = e.createLinearGradient(t[0], t[1], t[0], t[1] + t[3]);
                return n.addColorStop(0, "#ccc"), n.addColorStop(.5, "#fff"), n.addColorStop(.9, "#fff"), n.addColorStop(1, "#555"), n
            },
            leafNodeBodyLabeller: function (e, t, n, o) {
                e.rect(t[0], t[1], t[2], t[3]), e.clip(), e.fillStyle = TreemapUtils.avgRgb(n) <= 200 ? "#fff" : "#888", e.font = "0.625em Verdana, Geneva, sans-serif", e.fillText(o, t[0], t[1] + 10)
            },
            leafNodeBodyGradient: function (e, t, n) {
                var o = .1 * Math.min(t[2], t[3]), i = Math.max(t[2], t[3]), r = t[0] + .5 * t[2], s = t[1] + .5 * t[3],
                    a = e.createRadialGradient(r, s, o, r, s, i);
                return a.addColorStop(0, TreemapUtils.lighterColor(TreemapUtils.rgb2hex(n), .2)), a.addColorStop(1, TreemapUtils.darkerColor(TreemapUtils.rgb2hex(n), .2)), a
            },
            layoutMethod: TreemapUtils.squarify,
            sizeOption: 0,
            colorOption: 0,
            nodeBorderWidth: 0,
            labelsEnabled: !1,
            animationEnabled: !1,
            animationDurationMs: 1e3,
            animationEasing: {},
            postProcessCurve: {},
            nodeData: {}
        }, _create: function () {
            this.stats = {}
        }, _init: function () {
            this.stats = {}, this._refreshCanvas(), this._refreshColorGradient(), this._refreshColor(), this._refreshLayout(), this._renderNodes(), this._renderNodeLabels()
        }, _setOption: function (t) {
            switch (e.Widget.prototype._setOption.apply(this, arguments), t) {
                case"dimensions":
                    this._refreshCanvas(), this._refreshLayout(), this._renderNodes(), this._renderNodeLabels();
                    break;
                case"layoutMethod":
                    this._refreshLayout(), this._renderNodes(), this._renderNodeLabels();
                    break;
                case"labelsEnabled":
                    this._renderNodes(), this._renderNodeLabels();
                    break;
                case"leafNodeBodyGradient":
                    this._renderNodes(), this._renderNodeLabels();
                    break;
                case"colorStops":
                    this._refreshColorGradient(), this._refreshColor(), this._renderNodes(), this._renderNodeLabels();
                    break;
                case"colorOption":
                    this._refreshColor(), this._renderNodes(), this._renderNodeLabels();
                    break;
                case"sizeOption":
                    this._refreshLayout(), this._animateOptionChange();
                    break;
                case"postProcessCurve":
                    this._refreshLayout(), this._animateOptionChange();
                    break;
                case"nodeData":
                    this._refreshColor(), this._refreshLayout(), this._renderNodes(), this._renderNodeLabels()
            }
        }, _animateOptionChange: function () {
            function e(i) {
                var i = Date.now(), r = i - n;
                r < o.options.animationDurationMs ? (o._animateNodes(t.get(r / o.options.animationDurationMs)), requestAnimationFrame(e)) : (o._animationActive = !1, o._renderNodes(), o._renderNodeLabels())
            }

            if (1 == this.options.animationEnabled) {
                !function () {
                    var e = window.requestAnimationFrame || window.mozRequestAnimationFrame || window.webkitRequestAnimationFrame || window.msRequestAnimationFrame || function (e) {
                        window.setTimeout(e, 1e3 / 60)
                    };
                    window.requestAnimationFrame = e
                }(), this._animationActive = !0;
                var t = new TreemapUtils.KeySpline(this.options.animationEasing), n = Date.now(), o = this;
                requestAnimationFrame(e)
            } else this._renderNodes(), this._renderNodeLabels()
        }, _animateNodes: function (e) {
            var t = function (o) {
                var r, s, a;
                for (s = 0; s < o.length; s++) if (o[s].hasOwnProperty("children") === !1) {
                    if (void 0 !== o[s].prevGeometry) for (r = o[s].prevGeometry.body.slice(), a = 0; 4 > a; a++) r[a] += (o[s].geometry.body[a] - o[s].prevGeometry.body[a]) * e; else r = o[s].geometry.body;
                    i.save(), i.fillStyle = n.options.leafNodeBodyGradient.call(n, i, r, o[s].computedColor), i.fillRect(r[0], r[1], r[2], r[3]), i.restore()
                } else t(o[s].children)
            }, n = this, o = n.element.find("canvas")[0], i = o.getContext("2d");
            i.clearRect(0, 0, o.width, o.height), t([n.options.nodeData])
        }, _renderNodes: function () {
            if (void 0 === this._animationActive || this._animationActive !== !0) {
                var e = function (n) {
                    var o, r, s, a, l;
                    for (a = 0; a < n.length; a++) {
                        if (t._isRootNode(n[a]) === !1) {
                            if (o = n[a].geometry.body, r = n[a].geometry.header, s = o.slice(), o[2] <= 0 || o[3] <= 0) continue;
                            for (i.save(), n[a].hasOwnProperty("children") && null !== r ? (i.fillStyle = t.options.innerNodeHeaderGradient.call(t, i, r, n[a].computedColor), i.fillRect(r[0], r[1], r[2], r[3]), n[a].hasOwnProperty("children") && 0 === t.options.nodeBorderWidth && (i.strokeStyle = "#000", i.lineWidth = .5, i.beginPath(), i.moveTo(r[0] + r[2], r[1]), i.lineTo(r[0] + r[2], r[1] + r[3]), i.closePath(), i.stroke()), s[0] = r[0], s[1] = r[1], s[3] = r[3] + o[3]) : (i.fillStyle = t.options.leafNodeBodyGradient.call(t, i, o, n[a].computedColor), i.fillRect(o[0], o[1], o[2], o[3])), i.restore(), l = 0; l < s[3]; l++) t._addRunlength(s[0], s[0] + s[2], s[1] + l, n[a].id)
                        }
                        n[a].hasOwnProperty("children") && e(n[a].children)
                    }
                }, t = this, n = new Date, o = t.element.find("canvas")[0], i = o.getContext("2d");
                i.clearRect(0, 0, o.width, o.height), t._clearScanLines(), e([t.options.nodeData]);
                var r = new Date;
                t.stats.renderLayoutMsec = r - n
            }
        }, _renderNodeLabels: function () {
            if ((void 0 === this._animationActive || this._animationActive !== !0) && this.options.labelsEnabled === !0) {
                var e = function (n) {
                    var o, r, s;
                    for (s = 0; s < n.length; s++) {
                        if (t._isRootNode(n[s]) === !1) {
                            if (o = n[s].geometry.body, r = n[s].geometry.header, o[2] <= 0 || o[3] <= 0) {
                                n[s].hasOwnProperty("children") === !1 && (t.stats.leafNodeCnt += 1);
                                continue
                            }
                            i.save(), i.beginPath(), n[s].hasOwnProperty("children") ? null !== r && t.options.innerNodeHeaderLabeller.call(t, i, r, n[s].computedColor, n[s].id) : (t.options.leafNodeBodyLabeller.call(t, i, o, n[s].computedColor, n[s].id), t.stats.leafNodeCnt += 1, t.stats.renderedLeafNodeCnt += 1), i.restore()
                        }
                        n[s].hasOwnProperty("children") && e(n[s].children)
                    }
                }, t = this;
                t.stats.leafNodeCnt = 0, t.stats.renderedLeafNodeCnt = 0;
                var n = new Date, o = t.element.find("canvas")[0], i = o.getContext("2d");
                e([t.options.nodeData]);
                var r = new Date;
                t.stats.renderLabelsMsec = r - n
            }
        }, _refreshCanvas: function () {
            var e = this.element.find("canvas");
            e && e.remove(), e = document.createElement("canvas"), e.setAttribute("width", this.options.dimensions[0]), e.setAttribute("height", this.options.dimensions[1]);
            var t = this;
            this.element.append(e).mousemove(function (e) {
                if (void 0 === t._animationActive || t._animationActive !== !0) {
                    var n, o, i, r, s = t.element.offset(), a = parseInt(s.left, 10), l = parseInt(s.top, 10),
                        d = t.options.dimensions[0], h = t.options.dimensions[1];
                    if (e.pageX < a + d && e.pageY < l + h) {
                        for (n = t._coordsToId(e.pageX - a, e.pageY - l), o = [], r = 0; r < n.length; r++) o.push(t._getNode([n[r]]));
                        i = {nodes: o, ids: n}, t._trigger("mousemove", e, i)
                    }
                }
            }).click(function (e) {
                if (void 0 === t._animationActive || t._animationActive !== !0) {
                    var n, o, i, r, s = t.element.offset(), a = parseInt(s.left, 10), l = parseInt(s.top, 10),
                        d = t.options.dimensions[0], h = t.options.dimensions[1];
                    if (e.pageX < a + d && e.pageY < l + h) {
                        for (n = t._coordsToId(e.pageX - a, e.pageY - l), o = [], r = 0; r < n.length; r++) o.push(t._getNode([n[r]]));
                        i = {nodes: o, ids: n}, t._trigger("click", e, i)
                    }
                }
            })
        }, _refreshColorGradient: function () {
            var e, t, n, o = document.createElement("canvas"), i = this.options.colorStops;
            for (o.setAttribute("width", this.options.colorResolution), o.setAttribute("height", 1), "undefined" != typeof G_vmlCanvasManager && G_vmlCanvasManager.initElement(o), e = o.getContext("2d"), t = e.createLinearGradient(0, 0, this.options.colorResolution, 0), n = 0; n < i.length; n += 1) t.addColorStop(i[n].val, i[n].color);
            e.fillStyle = t, e.fillRect(0, 0, this.options.colorResolution, 1), this.options.colorGradientMap = e.getImageData(0, 0, this.options.colorResolution, 1)
        }, _refreshColor: function () {
            function e(n) {
                var o;
                for (o = 0; o < n.length; o++) void 0 !== n[o].color && (n[o].colorVal = n[o].color[t.options.colorOption], n[o].computedColor = t._getRgbColor(n[o].colorVal)), n[o].hasOwnProperty("children") && e(n[o].children)
            }

            var t = this;
            e([t.options.nodeData])
        }, _refreshLayout: function () {
            function e(e) {
                if (0 !== Object.getOwnPropertyNames(n.options.postProcessCurve).length) {
                    var t;
                    for (t = 0; t < e.length; t++) e[t] = i.get(e[t]);
                    var o = TreemapUtils.sumArray(e);
                    for (t = 0; t < e.length; t++) e[t] = e[t] / o
                }
                return e
            }

            function t(o, i) {
                var r, s, a, l, d = [];
                for (i.sort(function (e, t) {
                    return e.size[n.options.sizeOption] > t.size[n.options.sizeOption] ? -1 : e.size[n.options.sizeOption] < t.size[n.options.sizeOption] ? 1 : 0
                }), l = 0; l < i.length; l++) d[l] = n._isRootNode(i[l]) === !0 ? 1 : i[l].size[n.options.sizeOption];
                for (e(d), a = n.options.layoutMethod([o[0], o[1], o[2], o[3]], d), l = 0; l < i.length; l++) i[l].geometry && (i[l].prevGeometry = TreemapUtils.deepCopy(i[l].geometry)), i[l].geometry = {
                    body: a[l],
                    header: null
                }, n._addNode2NodeList(i[l]);
                for (l = 0; l < i.length; l++) i[l].hasOwnProperty("children") && (r = i[l].geometry.body, s = i[l].geometry.header, n._isRootNode(i[l]) === !1 && r[3] - n.options.innerNodeHeaderHeightPx > 0 && (s = i[l].geometry.header = r.slice(), s[3] = n.options.innerNodeHeaderHeightPx, r[1] += n.options.innerNodeHeaderHeightPx, r[3] -= n.options.innerNodeHeaderHeightPx), n._isRootNode(i[l]) === !1 && r[2] - n.options.nodeBorderWidth > 0 && r[3] - n.options.nodeBorderWidth > 0 && (n.options.dimensions[0] > r[0] + r[2] && (r[2] -= n.options.nodeBorderWidth, null !== s && (s[2] -= n.options.nodeBorderWidth)), n.options.dimensions[1] > r[1] + r[3] && (r[3] -= n.options.nodeBorderWidth)), t(r, i[l].children))
            }

            var n = this, o = new Date, i = new TreemapUtils.KeySpline(this.options.postProcessCurve),
                r = [0, 0, n.options.dimensions[0], n.options.dimensions[1]];
            n._clearNodeList(), t(r, [n.options.nodeData]);
            var s = new Date;
            n.stats.computeLayoutMsec = s - o
        }, _getRgbColor: function (e) {
            if (null === e) return TreemapUtils.hex2rgb(this.options.naColor);
            var t = this.options.colorGradientMap.data, n = 4 * Math.floor(e * (t.length / 4 - 1));
            return [t[n], t[n + 1], t[n + 2]]
        }, _clearScanLines: function () {
            this.scanLines && (this.scanLines.length = 0, this.scanLines = [])
        }, _addRunlength: function (e, t, n, o) {
            void 0 === this.scanLines && (this.scanLines = []), y_str = parseInt(n, 10), this.scanLines[y_str] || (this.scanLines[y_str] = []), this.scanLines[y_str].push([e, t, o])
        }, _coordsToId: function (e, t) {
            var n, o, i, r, s, a, l;
            if (void 0 === this.scanLines) return [];
            if (n = this.scanLines[t], s = [], n) for (l = n.length - 1; l >= 0; l--) o = n[l], i = o[0], r = o[1], a = o[2], e >= i && r > e && s.push(a);
            return s
        }, _clearNodeList: function () {
            this.nodeList && (this.nodeList = {})
        }, _addNode2NodeList: function (e) {
            void 0 === this.nodeList && (this.nodeList = {}), this.nodeList[e.id] || (this.nodeList[e.id] = e)
        }, _getNode: function (e) {
            return this.nodeList[e]
        }, _isRootNode: function (e) {
            return this.options.nodeData === e ? !0 : !1
        }, destroy: function () {
            this.element.find("canvas").remove(), e(window).unbind("resize"), e.Widget.prototype.destroy.call(this)
        }
    })
}(jQuery), String.prototype.trim = function () {
    return this.replace(/^\s+|\s+$/g, "")
}, String.prototype.ltrim = function () {
    return this.replace(/^\s+/, "")
}, String.prototype.rtrim = function () {
    return this.replace(/\s+$/, "")
}, String.prototype.lpad = function (e, t) {
    for (var n = this; n.length < t;) n = e + n;
    return n
}, String.prototype.rpad = function (e, t) {
    for (var n = this; n.length < t;) n += e;
    return n
};