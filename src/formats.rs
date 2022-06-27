/*
MIT License

Copyright (c) 2022 Siandfrance

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


use crate::{RED, GREEN, CLEAR};


pub fn format_bool(b: bool) -> String {
    if b {
        format!("{}true{}", GREEN, CLEAR)
    } else {
        format!("{}false{}", RED, CLEAR)
    }
}


pub fn format_unit(value: f32, base: f32, unit: &'static str) -> String {
    if value >= base.powi(8) {
        let val = value as f32 / base.powi(8);
        format!("{:.3} Y{}", val, unit)
    } else if value >= base.powi(7) {
        let val = value as f32 / base.powi(7);
        format!("{:.3} Z{}", val, unit)
    } else if value >= base.powi(6) {
        let val = value as f32 / base.powi(6);
        format!("{:.3} E{}", val, unit)
    } else if value >= base.powi(5) {
        let val = value as f32 / base.powi(5);
        format!("{:.3} P{}", val, unit)
    } else if value >= base.powi(4) {
        let val = value as f32 / base.powi(4);
        format!("{:.3} T{}", val, unit)
    } else if value >= base.powi(3) {
        let val = value as f32 / base.powi(3);
        format!("{:.3} G{}", val, unit)
    } else if value >= base.powi(2) {
        let val = value as f32 / base.powi(2);
        format!("{:.3} M{}", val, unit)
    } else if value >= base.powi(1) {
        let val = value as f32 / base.powi(1);
        format!("{:.3} K{}", val, unit)
    } else {
        format!("{} {}", value, unit)
    }
}


pub fn format_mem(mem: u64) -> String {
    format_unit(mem as f32 / 8f32, 1024f32, "iB")
}


pub fn format_freq(freq: f32) -> String {
    format_unit(freq, 1000f32, "Hz")
}