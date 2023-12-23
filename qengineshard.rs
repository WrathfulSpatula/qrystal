use std::collections::HashMap;
use std::rc::Rc;

struct PhaseShard {
    cmplxDiff: Complex,
    cmplxSame: Complex,
    isInvert: bool,
}

impl PhaseShard {
    fn new() -> Self {
        PhaseShard {
            cmplxDiff: ONE_CMPLX,
            cmplxSame: ONE_CMPLX,
            isInvert: false,
        }
    }
}

struct QEngineShard {
    unit: Option<Rc<dyn QInterface>>,
    mapped: bitLenInt,
    isProbDirty: bool,
    isPhaseDirty: bool,
    amp0: Complex,
    amp1: Complex,
    pauliBasis: Pauli,
    controlsShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    antiControlsShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    targetOfShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    antiTargetOfShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    found: bool,
}

impl QEngineShard {
    fn new() -> Self {
        QEngineShard {
            unit: None,
            mapped: 0,
            isProbDirty: false,
            isPhaseDirty: false,
            amp0: ONE_CMPLX,
            amp1: ZERO_CMPLX,
            pauliBasis: PauliZ,
            controlsShards: HashMap::new(),
            antiControlsShards: HashMap::new(),
            targetOfShards: HashMap::new(),
            antiTargetOfShards: HashMap::new(),
            found: false,
        }
    }

    fn new_with_set(set: bool, rand_phase: Complex) -> Self {
        let mut shard = QEngineShard::new();
        shard.amp0 = if set { ZERO_CMPLX } else { rand_phase };
        shard.amp1 = if set { rand_phase } else { ZERO_CMPLX };
        shard
    }

    fn new_with_unit(u: Rc<dyn QInterface>, mapping: bitLenInt) -> Self {
        QEngineShard {
            unit: Some(u),
            mapped: mapping,
            isProbDirty: true,
            isPhaseDirty: true,
            amp0: ONE_CMPLX,
            amp1: ZERO_CMPLX,
            pauliBasis: PauliZ,
            controlsShards: HashMap::new(),
            antiControlsShards: HashMap::new(),
            targetOfShards: HashMap::new(),
            antiTargetOfShards: HashMap::new(),
            found: false,
        }
    }

    fn make_dirty(&mut self) {
        self.isProbDirty = true;
        self.isPhaseDirty = true;
    }

    fn clamp_amps(&mut self) -> bool {
        // TODO: Implement this function
        false
    }

    fn dump_multi_bit(&self) {
        // TODO: Implement this function
    }

    fn remove_buffer(
        &mut self,
        p: Rc<QEngineShard>,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_map_get: fn(&QEngineShard) -> &HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    ) {
        // TODO: Implement this function
    }

    fn remove_control(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.targetOfShards, QEngineShard::get_controls_shards);
    }

    fn remove_target(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.controlsShards, QEngineShard::get_target_of_shards);
    }

    fn remove_anti_control(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.antiTargetOfShards, QEngineShard::get_anti_controls_shards);
    }

    fn remove_anti_target(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.antiControlsShards, QEngineShard::get_anti_target_of_shards);
    }

    fn dump_buffer(
        &mut self,
        optimize_fn: fn(&mut QEngineShard),
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        // TODO: Implement this function
    }

    fn dump_same_phase_buffer(
        &mut self,
        optimize_fn: fn(&mut QEngineShard),
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        // TODO: Implement this function
    }

    fn dump_control_of(&mut self) {
        self.dump_buffer(
            QEngineShard::optimize_targets,
            &mut self.controlsShards,
            QEngineShard::remove_target,
        );
    }

    fn dump_anti_control_of(&mut self) {
        self.dump_buffer(
            QEngineShard::optimize_anti_targets,
            &mut self.antiControlsShards,
            QEngineShard::remove_anti_target,
        );
    }

    fn dump_same_phase_control_of(&mut self) {
        self.dump_same_phase_buffer(
            QEngineShard::optimize_targets,
            &mut self.controlsShards,
            QEngineShard::remove_target,
        );
    }

    fn dump_same_phase_anti_control_of(&mut self) {
        self.dump_same_phase_buffer(
            QEngineShard::optimize_anti_targets,
            &mut self.antiControlsShards,
            QEngineShard::remove_anti_target,
        );
    }

    fn add_buffer(
        &mut self,
        p: Rc<QEngineShard>,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        // TODO: Implement this function
    }

    fn make_phase_controlled_by(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.targetOfShards, QEngineShard::get_controls_shards);
    }

    fn make_phase_control_of(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.controlsShards, QEngineShard::get_target_of_shards);
    }

    fn make_phase_anti_controlled_by(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.antiTargetOfShards, QEngineShard::get_anti_controls_shards);
    }

    fn make_phase_anti_control_of(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.antiControlsShards, QEngineShard::get_anti_target_of_shards);
    }

    fn add_angles(
        &mut self,
        control: Rc<QEngineShard>,
        cmplxDiff: Complex,
        cmplxSame: Complex,
        local_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        // TODO: Implement this function
    }

    fn add_phase_angles(&mut self, control: Rc<QEngineShard>, topLeft: Complex, bottomRight: Complex) {
        self.add_angles(
            control,
            topLeft,
            bottomRight,
            QEngineShard::make_phase_controlled_by,
            &mut self.targetOfShards,
            QEngineShard::remove_control,
        );
    }

    fn add_anti_phase_angles(&mut self, control: Rc<QEngineShard>, bottomRight: Complex, topLeft: Complex) {
        self.add_angles(
            control,
            bottomRight,
            topLeft,
            QEngineShard::make_phase_anti_controlled_by,
            &mut self.antiTargetOfShards,
            QEngineShard::remove_anti_control,
        );
    }

    fn add_inversion_angles(&mut self, control: Rc<QEngineShard>, topRight: Complex, bottomLeft: Complex) {
        self.make_phase_controlled_by(control);
        self.targetOfShards[&control].isInvert = !self.targetOfShards[&control].isInvert;
        std::mem::swap(
            &mut self.targetOfShards[&control].cmplxDiff,
            &mut self.targetOfShards[&control].cmplxSame,
        );
        self.add_phase_angles(control, topRight, bottomLeft);
    }

    fn add_anti_inversion_angles(&mut self, control: Rc<QEngineShard>, bottomLeft: Complex, topRight: Complex) {
        self.make_phase_anti_controlled_by(control);
        self.antiTargetOfShards[&control].isInvert = !self.antiTargetOfShards[&control].isInvert;
        std::mem::swap(
            &mut self.antiTargetOfShards[&control].cmplxDiff,
            &mut self.antiTargetOfShards[&control].cmplxSame,
        );
        self.add_anti_phase_angles(control, bottomLeft, topRight);
    }

    fn optimize_buffer(
        &mut self,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_map_get: fn(&QEngineShard) -> &HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        phase_fn: fn(&mut QEngineShard, Rc<QEngineShard>, Complex, Complex),
        make_this_control: bool,
    ) {
        // TODO: Implement this function
    }

    fn optimize_controls(&mut self) {
        self.optimize_buffer(
            &mut self.controlsShards,
            QEngineShard::get_target_of_shards,
            QEngineShard::add_phase_angles,
            false,
        );
    }

    fn optimize_targets(&mut self) {
        self.optimize_buffer(
            &mut self.targetOfShards,
            QEngineShard::get_controls_shards,
            QEngineShard::add_phase_angles,
            true,
        );
    }

    fn optimize_anti_controls(&mut self) {
        self.optimize_buffer(
            &mut self.antiControlsShards,
            QEngineShard::get_anti_target_of_shards,
            QEngineShard::add_anti_phase_angles,
            false,
        );
    }

    fn optimize_anti_targets(&mut self) {
        self.optimize_buffer(
            &mut self.antiTargetOfShards,
            QEngineShard::get_anti_controls_shards,
            QEngineShard::add_anti_phase_angles,
            true,
        );
    }

    fn optimize_both_targets(&mut self) {
        // TODO: Implement this function
    }

    fn combine_buffers(
        &mut self,
        target_map_get: fn(&QEngineShard) -> &HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        control_map_get: fn(&QEngineShard) -> &HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        angle_fn: fn(&mut QEngineShard, Rc<QEngineShard>, Complex, Complex),
    ) {
        // TODO: Implement this function
    }

    fn combine_gates(&mut self) {
        self.combine_buffers(
            QEngineShard::get_target_of_shards,
            QEngineShard::get_controls_shards,
            QEngineShard::add_phase_angles,
        );
        self.combine_buffers(
            QEngineShard::get_anti_target_of_shards,
            QEngineShard::get_anti_controls_shards,
            QEngineShard::add_anti_phase_angles,
        );
    }

    fn swap_target_anti(&mut self, control: Rc<QEngineShard>) {
        // TODO: Implement this function
    }

    fn flip_phase_anti(&mut self) {
        // TODO: Implement this function
    }

    fn commute_phase(&mut self, topLeft: Complex, bottomRight: Complex) {
        // TODO: Implement this function
    }

    fn remove_identity_buffers(
        &mut self,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_map_get: fn(&QEngineShard) -> &HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    ) {
        // TODO: Implement this function
    }

    fn remove_phase_buffers(
        &mut self,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_map_get: fn(&QEngineShard) -> &HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    ) {
        // TODO: Implement this function
    }

    fn commute_h(&mut self) {
        self.remove_phase_buffers(&mut self.targetOfShards, QEngineShard::get_controls_shards);
        self.remove_phase_buffers(&mut self.antiTargetOfShards, QEngineShard::get_anti_controls_shards);
        self.remove_phase_buffers(&mut self.controlsShards, QEngineShard::get_target_of_shards);
        self.remove_phase_buffers(&mut self.antiControlsShards, QEngineShard::get_anti_target_of_shards);
    }

    fn dump_phase_buffers(&mut self) {
        self.remove_phase_buffers(&mut self.targetOfShards, QEngineShard::get_controls_shards);
        self.remove_phase_buffers(&mut self.antiTargetOfShards, QEngineShard::get_anti_controls_shards);
        self.remove_phase_buffers(&mut self.controlsShards, QEngineShard::get_target_of_shards);
        self.remove_phase_buffers(&mut self.antiControlsShards, QEngineShard::get_anti_target_of_shards);
    }

    fn is_invert_control(&self) -> bool {
        // TODO: Implement this function
        false
    }

    fn is_invert_target(&self) -> bool {
        // TODO: Implement this function
        false
    }

    fn clear_map_invert_phase(&mut self, shards: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>) {
        // TODO: Implement this function
    }

    fn clear_invert_phase(&mut self) {
        self.clear_map_invert_phase(&mut self.controlsShards);
        self.clear_map_invert_phase(&mut self.antiControlsShards);
        self.clear_map_invert_phase(&mut self.targetOfShards);
        self.clear_map_invert_phase(&mut self.antiTargetOfShards);
    }

    fn get_qubit_count(&self) -> bitLenInt {
        self.unit.as_ref().map_or(1, |u| u.get_qubit_count())
    }

    fn prob(&self) -> real1_f {
        if !self.isProbDirty || self.unit.is_none() {
            return amp1.norm();
        }
        self.unit.as_ref().unwrap().prob(mapped)
    }

    fn is_clifford(&self) -> bool {
        if let Some(unit) = &self.unit {
            unit.is_clifford(mapped)
        } else {
            amp0.norm() <= FP_NORM_EPSILON
                || amp1.norm() <= FP_NORM_EPSILON
                || (amp0 - amp1).norm() <= FP_NORM_EPSILON
                || (amp0 + amp1).norm() <= FP_NORM_EPSILON
                || (amp0 - I_CMPLX * amp1).norm() <= FP_NORM_EPSILON
                || (amp0 + I_CMPLX * amp1).norm() <= FP_NORM_EPSILON
        }
    }
}

struct QEngineShardMap {
    shards: Vec<QEngineShard>,
    swapMap: Vec<bitLenInt>,
}

impl QEngineShardMap {
    fn new() -> Self {
        QEngineShardMap {
            shards: Vec::new(),
            swapMap: Vec::new(),
        }
    }

    fn new_with_size(size: bitLenInt) -> Self {
        let mut map = QEngineShardMap::new();
        for i in 0..size {
            map.swapMap.push(i);
        }
        map
    }

    fn push_back(&mut self, shard: QEngineShard) {
        self.shards.push(shard);
        self.swapMap.push(self.swapMap.len() as bitLenInt);
    }

    fn insert(&mut self, start: bitLenInt, to_insert: &mut QEngineShardMap) {
        let o_size = self.size();
        self.shards.extend(to_insert.shards.drain(..));
        self.swapMap.splice(start..start, to_insert.swapMap.drain(..));
        for lcv in 0..to_insert.size() {
            self.swapMap[start + lcv] += o_size;
        }
    }

    fn erase(&mut self, begin: bitLenInt, end: bitLenInt) {
        for index in begin..end {
            let offset = self.swapMap[index];
            self.shards.remove(offset as usize);
            for lcv in 0..self.swapMap.len() {
                if self.swapMap[lcv] >= offset {
                    self.swapMap[lcv] -= 1;
                }
            }
        }
        self.swapMap.drain(begin..end);
    }

    fn swap(&mut self, qubit1: bitLenInt, qubit2: bitLenInt) {
        self.swapMap.swap(qubit1 as usize, qubit2 as usize);
    }
}


