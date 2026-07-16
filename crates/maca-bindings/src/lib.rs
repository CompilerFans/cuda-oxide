/*
 * MXMACA Runtime API bindings
 *
 * Auto-generated bindings to MetaX MXMACA Runtime API.
 */

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Check if a MXMACA runtime call succeeded.
pub fn check(result: mcError_t) -> Result<(), String> {
    if result == _mcError_t_mcSuccess as mcError_t {
        Ok(())
    } else {
        Err(format!("MXMACA error: {}", result))
    }
}
