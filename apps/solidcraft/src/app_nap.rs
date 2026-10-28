//! While a recording runs, macOS must not nap the app (App Nap throttles the timers and the
//! frames of a covered window): an `NSProcessInfo` activity holds it awake until it ends.

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};

#[derive(Default)]
pub struct AppNap {
    activity: Option<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
}

impl AppNap {
    /// Hold the app awake while `awake`, let it nap again once not.
    pub fn hold(&mut self, awake: bool) {
        match (awake, self.activity.is_some()) {
            (true, false) => {
                let reason = NSString::from_str("Recording frames");
                let options = NSActivityOptions::UserInitiated | NSActivityOptions::LatencyCritical;
                self.activity = Some(NSProcessInfo::processInfo().beginActivityWithOptions_reason(options, &reason));
            }
            (false, true) => {
                if let Some(activity) = self.activity.take() {
                    // SAFETY: `activity` is the object `beginActivityWithOptions:reason:` returned.
                    #[allow(unsafe_code)]
                    unsafe {
                        NSProcessInfo::processInfo().endActivity(&activity);
                    }
                }
            }
            _ => {}
        }
    }
}
