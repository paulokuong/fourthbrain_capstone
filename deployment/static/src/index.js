/*
Main page
=========
*/
'use strict';

import _ from 'lodash';
import axios from 'axios';
import dateformat from 'dateformat';
import 'bootstrap';
import 'bootstrap/js/dist/tooltip';


class Index {
    constructor(env) {

    }
    hookMouseEvents() {

    }

}

$(document).ready(function(e) {
    var i = new Index();
    i.hookMouseEvents();
    $('[data-toggle="tooltip"]').tooltip();
    $('#carousel').carousel({
        interval: 2000
    })
});